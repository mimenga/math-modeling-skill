# -*- coding: utf-8 -*-
"""
12 C题定价补货专题（数据题高分进阶工具箱）
用途：价格弹性回归 / 内生性 2SLS / 分位数回归 / 报童补货 / STL 分解 / 帕累托 ABC
接口约定：DataFrame / array 进，dict / DataFrame 出——返回值可直接写入 results.json
     供论文动态引用（见 SKILL.md 硬约束第 9 条）。
依赖：numpy, pandas, scipy, statsmodels
对照卡片：references/03_统计与数据挖掘/06_内生性工具变量与分位数回归.md
         references/02_建模方法库/06_报童模型与随机库存优化.md
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


# ---------- 1. 价格弹性回归（log-log OLS） ----------
def elasticity_regression(df, price_col, qty_col, controls=None):
    """log(销量) = b0 + b1*log(价格) + 控制项 的 OLS，b1 即价格弹性。

    参数：df 需含价格列与销量列（取对数的行要求 >0，自动剔除 0/负值）；
         controls 为控制变量列名列表（如促销标记、节假日）。
    返回 dict：elasticity / se / t / p / r2 / n（论文直接引用）。
    """
    d = df[[price_col, qty_col]].astype(float).copy()
    d = d[(d[price_col] > 0) & (d[qty_col] > 0)]           # 对数保护
    y = np.log(d[qty_col]).values
    X = pd.DataFrame({"log_price": np.log(d[price_col]).values})
    if controls:
        X = pd.concat([X, df.loc[d.index, controls].astype(float).reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    res = sm.OLS(y, X).fit()
    return {"elasticity": float(res.params.iloc[1]), "se": float(res.bse.iloc[1]),
            "t": float(res.tvalues.iloc[1]), "p": float(res.pvalues.iloc[1]),
            "r2": float(res.rsquared), "n": int(res.nobs)}


# ---------- 2. 工具变量 2SLS 弹性（处理价格-销量内生性） ----------
def iv2sls_elasticity(df, price_col, qty_col, iv_col, controls=None):
    """批发成本作工具变量的两阶段最小二乘弹性。

    第一阶段：log(价格) ~ 工具变量 + 控制项（报告 iv 的 F = t²，>10 才非弱工具）；
    第二阶段：log(销量) ~ 价格拟合值 + 控制项，系数即因果弹性估计。
    返回 dict：elasticity / se / p / first_f / n。
    """
    d = df[[price_col, qty_col, iv_col]].astype(float).copy()
    d = d[(d[price_col] > 0) & (d[qty_col] > 0) & (d[iv_col] > 0)]
    y = np.log(d[qty_col]).values
    ctrl = (df.loc[d.index, controls].astype(float).reset_index(drop=True)
            if controls else pd.DataFrame(index=range(len(d))))

    X1 = pd.DataFrame({"iv": np.log(d[iv_col]).values})
    X1 = sm.add_constant(pd.concat([X1, ctrl], axis=1))
    first = sm.OLS(np.log(d[price_col]).values, X1).fit()
    p_hat = first.fittedvalues
    first_f = float(first.tvalues.iloc[1] ** 2)            # 单工具变量 F = t²

    X2 = pd.DataFrame({"p_hat": p_hat})
    X2 = sm.add_constant(pd.concat([X2, ctrl], axis=1))
    second = sm.OLS(y, X2).fit()
    return {"elasticity": float(second.params.iloc[1]), "se": float(second.bse.iloc[1]),
            "p": float(second.pvalues.iloc[1]), "first_f": first_f, "n": int(second.nobs)}


# ---------- 3. 分位数回归弹性（异方差/尾部/稳健） ----------
def quantile_elasticity(df, price_col, qty_col, taus=(0.25, 0.5, 0.75)):
    """各分位数水平下的价格弹性，返回 DataFrame(tau, elasticity, se, p)。

    论文亮点句式：弹性随 tau 的变化趋势 = 薄销/畅销端对价格的不同敏感度。
    """
    from statsmodels.regression.quantile_regression import QuantReg
    d = df[[price_col, qty_col]].astype(float).copy()
    d = d[(d[price_col] > 0) & (d[qty_col] > 0)]
    y = np.log(d[qty_col]).values
    X = sm.add_constant(pd.DataFrame({"log_price": np.log(d[price_col]).values}))
    rows = []
    for tau in taus:
        r = QuantReg(y, X).fit(q=tau)
        rows.append({"tau": tau, "elasticity": float(r.params.iloc[1]),
                     "se": float(r.bse.iloc[1]), "p": float(r.pvalues.iloc[1])})
    return pd.DataFrame(rows)


# ---------- 4. 报童模型（正态版 + 经验分位数版） ----------
def newsvendor_normal(mu, sigma, cu, co):
    """需求 ~ N(mu, sigma²) 的最优补货量。cu=缺货损失，co=过剩损失。

    返回 (Q*, 临界比)；临界比 = cu/(cu+co)。
    """
    if sigma <= 0 or cu <= 0 or co <= 0:
        raise ValueError("sigma/cu/co 必须为正")
    ratio = cu / (cu + co)
    return float(mu + sigma * stats.norm.ppf(ratio)), float(ratio)


def newsvendor_quantile(demand_samples, cu, co):
    """经验分位数版报童（无分布假设，评委问'凭什么正态'时的答案）。

    demand_samples：历史或预测+残差扰动得到的需求样本。
    返回 (Q*, 临界比)。
    """
    s = np.asarray(demand_samples, dtype=float)
    if s.size < 30:
        raise ValueError("需求样本 <30，经验分位数不稳，请改用正态版或补样本")
    if cu <= 0 or co <= 0:
        raise ValueError("cu/co 必须为正")
    ratio = cu / (cu + co)
    return float(np.quantile(s, ratio)), float(ratio)


# ---------- 5. STL 季节分解（销量序列去季节） ----------
def stl_decompose(series, period=7):
    """STL 稳健分解，返回 DataFrame(observed/trend/seasonal/resid)。

    去季节后的序列 = observed - seasonal，可喂给回归/预测（2023C 论文1 的去周期思路）。
    """
    from statsmodels.tsa.seasonal import STL
    s = np.asarray(series, dtype=float)
    if s.size < 2 * period:
        raise ValueError(f"序列长度 {s.size} 不足两倍周期 {period}，无法分解")
    out = STL(s, period=period, robust=True).fit()
    return pd.DataFrame({"observed": out.observed, "trend": out.trend,
                         "seasonal": out.seasonal, "resid": out.resid})


# ---------- 6. 帕累托 / ABC 分级（选品与重点管理） ----------
def pareto_abc(df, value_col, a_cut=0.8, b_cut=0.95):
    """按 value_col 降序做 ABC 分级，返回按贡献排序的 DataFrame（含累计占比与标签）。

    a_cut/b_cut 为累计贡献阈值（80%/95% 经典口径，可按题改）。
    """
    if df.empty or value_col not in df.columns:
        raise ValueError("df 为空或缺少价值列")
    d = df.sort_values(value_col, ascending=False).copy()
    total = d[value_col].sum()
    if total <= 0:
        raise ValueError("价值列合计非正，无法分级")
    d["share"] = d[value_col] / total
    d["cum_share"] = d["share"].cumsum()
    d["class"] = np.where(d["cum_share"] <= a_cut, "A",
                          np.where(d["cum_share"] <= b_cut, "B", "C"))
    return d


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n = 400
    price = rng.uniform(2, 8, n)
    qty = 100 * price ** -0.8 * np.exp(rng.normal(0, 0.1, n))     # 真弹性 -0.8
    cost = price / (1 + rng.uniform(0.2, 0.6, n))                 # 成本加成 → 工具变量
    promo = rng.integers(0, 2, n)
    df = pd.DataFrame({"price": price, "qty": qty, "cost": cost, "promo": promo})

    print("1) log-log 弹性：", {k: round(v, 4) for k, v in elasticity_regression(df, "price", "qty").items()})
    ols_e = elasticity_regression(df, "price", "qty", controls=["promo"])["elasticity"]
    iv = iv2sls_elasticity(df, "price", "qty", "cost")
    print(f"2) 含控制 OLS 弹性={ols_e:.4f}；2SLS 弹性={iv['elasticity']:.4f}"
          f"（第一阶段 F={iv['first_f']:.0f}）")
    print("3) 分位数弹性：\n", quantile_elasticity(df, "price", "qty").round(4).to_string(index=False))
    q_n, r1 = newsvendor_normal(120, 25, cu=3, co=1)
    q_e, r2 = newsvendor_quantile(rng.normal(120, 25, 5000), cu=3, co=1)
    print(f"4) 报童：正态版 Q*={q_n:.1f}，经验版 Q*={q_e:.1f}（临界比={r1:.3f}）")
    ts = np.arange(70) + 5 * np.sin(np.arange(70) * 2 * np.pi / 7) + rng.normal(0, .5, 70)
    stl = stl_decompose(ts, period=7)
    print(f"5) STL：输出 {stl.shape}，季节项幅度≈{stl['seasonal'].max() - stl['seasonal'].min():.1f}")
    abc = pareto_abc(pd.DataFrame({"item": [f"i{i}" for i in range(20)],
                                   "sales": rng.pareto(1.2, 20) * 100}), "sales")
    print(f"6) 帕累托：A 类 {int((abc['class'] == 'A').sum())} 个单品贡献 "
          f"{abc.loc[abc['class'] == 'A', 'cum_share'].max():.0%}")
