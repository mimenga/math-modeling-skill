# -*- coding: utf-8 -*-
"""
smoke_test.py 环境自检
用途：换电脑/新环境后一键验证所有代码模板能否跑通（赛前必做）。
运行：python smoke_test.py
说明：
  - 以 headless 模式运行（MPLBACKEND=Agg），不弹图窗；
  - 在临时目录中运行各脚本，不污染本目录；
  - MATLAB 脚本（matlab/ 目录）不在自检范围，需在 MATLAB 中单独验证；
  - 可选依赖（pulp / xgboost）未安装不算失败，脚本会自动跳过相应演示；
  - 第二阶段做函数级回归测试：覆盖 __main__ 演示触不到的边界输入
    （历史上熵权法常数列、max_flow 重复边等静默 bug 就是这样漏掉的）。
"""
import os
import sys
import glob
import tempfile
import traceback
from subprocess import run

os.environ["MPLBACKEND"] = "Agg"  # headless，防止 plt.show() 阻塞

SCRIPTS = sorted(p for p in glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
                 if not os.path.basename(p).endswith("smoke_test.py"))


def _load_module(fname):
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), fname)
    spec = importlib.util.spec_from_file_location("reg_" + fname[:-3], path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def regression_checks():
    """函数级回归测试：每个用例对应一个历史上真实出现过的 bug。"""
    import warnings
    import numpy as np
    import pandas as pd

    m01 = _load_module("01_插值与拟合.py")
    m02 = _load_module("02_线性与整数规划.py")
    m03 = _load_module("03_回归与分类.py")
    m04 = _load_module("04_聚类.py")
    m05 = _load_module("05_时间序列.py")
    m06 = _load_module("06_评价方法.py")
    m08 = _load_module("08_图论.py")
    m09 = _load_module("09_数据处理与EDA.py")
    m10 = _load_module("10_可视化模板.py")
    m12 = _load_module("12_C题定价补货专题.py")
    m13 = _load_module("13_数字一致性校验.py")

    checks = []

    def add(name, fn):
        checks.append((name, fn))

    # ---- 06 熵权法：常数列权重必须为 0（曾静默拿到最大权重） ----
    def t_entropy_const_col():
        w = m06.entropy_weight([[1., 5., 2.], [2., 5., 3.], [3., 5., 1.], [4., 5., 4.]])
        assert abs(w[1]) < 1e-9 and abs(w.sum() - 1) < 1e-9, w
    add("06 熵权法：常数列权重=0", t_entropy_const_col)

    def t_entropy_single_row():
        w = m06.entropy_weight([[1., 2., 3.]])
        assert np.allclose(w, 1 / 3), w
    add("06 熵权法：单方案返回等权重", t_entropy_single_row)

    # ---- 06 TOPSIS：全 0 指标列不得产生 NaN ----
    def t_topsis_zero_col():
        s = m06.topsis([[8., 7., 0.], [6., 9., 0.], [7., 6., 0.]], [.4, .3, .3], [1, 1, 0])
        assert np.all(np.isfinite(s)), s
    add("06 TOPSIS：全0列不产生NaN", t_topsis_zero_col)

    # ---- 03 特征重要性：ndarray names 不崩、连续 y 自动用回归器 ----
    def t_feature_importance_ndarray():
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
        imp = m03.feature_importance(d.data, d.target, names=d.feature_names)
        assert len(imp) == d.data.shape[1]
    add("03 特征重要性：names 传 ndarray", t_feature_importance_ndarray)

    def t_feature_importance_continuous_y():
        from sklearn.datasets import load_diabetes
        d = load_diabetes()
        imp = m03.feature_importance(d.data, d.target, names=list(d.feature_names))
        assert len(imp) == d.data.shape[1]
    add("03 特征重要性：连续 y 自动用回归器", t_feature_importance_continuous_y)

    # ---- 01 插值：重复 x 不崩、lagrange 弃用 API 已替换 ----
    def t_interp_duplicate_x():
        x = [1., 2., 2., 3., 4.]
        y = [1., 3., 2.5, 4., 2.]
        out = m01.interp_demo(x, y, np.array([1.5, 2.5, 3.5]))
        assert all(np.all(np.isfinite(o)) for o in out)
    add("01 插值：重复 x 自动合并", t_interp_duplicate_x)

    # ---- 10 scatter_fit：重复 x 不崩、list 输入可用 ----
    def t_scatter_fit_duplicate_x():
        rng = np.random.default_rng(0)
        x = np.sort(rng.choice(np.arange(10.), 30))
        m10.scatter_fit(x, rng.normal(0, 1, 30))
        m10.scatter_fit([1, 2, 3, 4, 5], [2, 3, 5, 4, 6])
    add("10 scatter_fit：重复 x / list 输入", t_scatter_fit_duplicate_x)

    # ---- 02 规划：不可行要抛错而不是静默返回 None ----
    def t_lp_infeasible_raises():
        try:
            m02.lp_solve(np.array([-1., -1]), np.array([[1., 1], [-1, -1]]), np.array([1., -3.]))
        except ValueError:
            return
        raise AssertionError("不可行问题未抛错")
    add("02 线性规划：不可行抛 ValueError", t_lp_infeasible_raises)

    def t_milp_infeasible_raises():
        try:
            m02.milp_solve(np.array([-1., -1]), np.array([[1., 1], [-1, -1]]),
                           np.array([1., -3.]), np.array([1, 1]))
        except ValueError:
            return
        raise AssertionError("不可行问题未抛错")
    add("02 整数规划：不可行抛 ValueError", t_milp_infeasible_raises)

    # ---- 08 max_flow：重复边容量累加（曾直接覆盖） ----
    def t_max_flow_parallel_edges():
        v = m08.max_flow([(0, 1, 3), (0, 1, 4), (1, 2, 10)], 0, 2)
        assert v == 7, v
    add("08 最大流：重复边容量累加=7", t_max_flow_parallel_edges)

    # ---- 05 GM(1,1)：平稳/非正数据要抛错（曾输出垃圾预测） ----
    def t_gm11_flat_raises():
        try:
            m05.gm11([5, 5, 5, 5, 5], 1)
        except ValueError:
            return
        raise AssertionError("平稳序列未抛错")
    add("05 GM(1,1)：平稳序列抛 ValueError", t_gm11_flat_raises)

    def t_gm11_negative_raises():
        try:
            m05.gm11([10, -8, 6, 4, 2], 1)
        except ValueError:
            return
        raise AssertionError("含负值序列未抛错")
    add("05 GM(1,1)：含负值抛 ValueError", t_gm11_negative_raises)

    # ---- 09 handle_missing：不得修改调用者的 df ----
    def t_handle_missing_no_mutation():
        df = pd.DataFrame({"A": [1., 2., np.nan, 4.]})
        orig = df.copy()
        m09.handle_missing(df, "mean")
        assert df["A"].equals(orig["A"]), "handle_missing 原地修改了调用者的 df"
    add("09 缺失值处理：不修改原 df", t_handle_missing_no_mutation)

    # ---- 04 evaluate：DBSCAN 噪声点不计入评估 ----
    def t_evaluate_excludes_noise():
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, .3, (30, 2)), rng.normal(5, .3, (30, 2))])
        labels = np.array([0] * 30 + [1] * 30)
        labels[:5] = -1                       # 混入噪声点
        sil, ch = m04.evaluate(X, labels)
        assert sil is not None and np.isfinite(sil) and np.isfinite(ch)
    add("04 聚类评估：排除 -1 噪声点", t_evaluate_excludes_noise)

    # ---- 10 扩展：折线数值 x / 分组柱状 / 3D 色条 / 矩阵热图 / 残差诊断 / 散点矩阵 ----
    def t_line_chart_numeric_x():
        m10.line_chart({"A": np.arange(10.0), "B": np.arange(10.0)[::-1]},
                       x=np.linspace(0, 5, 10))
    add("10 折线图：支持数值 x 轴", t_line_chart_numeric_x)

    def t_bar_chart_grouped():
        import matplotlib.pyplot as plt
        m10.bar_chart(["a", "b", "c"], {"S1": [1, 2, 3], "S2": [3, 2, 1]}, stacked=True)
        # 堆叠图横轴必须是类别名而非 0,1,2…（曾静默丢失标签）
        labels = [t.get_text() for t in plt.gca().get_xticklabels()]
        assert any(l in labels for l in ("a", "b", "c")), labels
    add("10 柱状图：分组/堆叠 + 横轴类别名", t_bar_chart_grouped)

    def t_surface_3d_colorbar():
        xx, yy = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
        m10.surface_3d(xx, yy, xx * yy, cbar_label="z")
    add("10 三维曲面：支持 colorbar", t_surface_3d_colorbar)

    def t_matrix_heatmap():
        zz = np.random.default_rng(0).normal(size=(20, 30)).cumsum(axis=1)
        m10.matrix_heatmap(zz, cbar_label="强度", xlabel="x", ylabel="y")
    add("10 矩阵着色热力图", t_matrix_heatmap)

    def t_residual_qq():
        rng = np.random.default_rng(1)
        yt = np.linspace(0, 10, 40) + rng.normal(0, 0.3, 40)
        yp = np.linspace(0, 10, 40)
        m10.residual_plot(yt, yp)
        osm, osr, r2 = m10.qq_plot(yt - yp)
        assert 0 <= r2 <= 1
    add("10 残差诊断：residual_plot + qq_plot", t_residual_qq)

    def t_pair_matrix():
        import pandas as pd
        df = pd.DataFrame(np.random.default_rng(2).normal(size=(50, 3)),
                          columns=list("ABC"))
        m10.pair_matrix(df)
    add("10 散点矩阵 EDA", t_pair_matrix)

    # ---- 12 弹性回归：合成数据（真弹性 -0.8）符号与显著性 ----
    def t12_elasticity_sign():
        rng = np.random.default_rng(7)
        p = rng.uniform(2, 8, 400)
        q = 100 * p ** -0.8 * np.exp(rng.normal(0, .1, 400))
        r = m12.elasticity_regression(pd.DataFrame({"p": p, "q": q}), "p", "q")
        assert -1.0 < r["elasticity"] < -0.6 and r["p"] < 1e-10, r
    add("12 弹性回归：合成数据 β≈-0.8 且显著", t12_elasticity_sign)

    # ---- 12 2SLS：内生数据下须比 OLS 更接近真弹性 -0.9 ----
    def t12_2sls_closer_than_ols():
        rng = np.random.default_rng(11)
        n = 600
        shock = rng.normal(0, 1, n)
        logc = rng.uniform(0.0, 1.2, n)                # 工具变量：log 批发成本（外生）
        logp = logc + 0.8 * shock + rng.normal(0, .15, n)   # 成本加成定价（混入需求冲击）
        lq = 3 - 0.9 * logp + 0.5 * shock + rng.normal(0, .2, n)  # 真弹性 -0.9
        df = pd.DataFrame({"p": np.exp(logp), "q": np.exp(lq), "cost": np.exp(logc)})
        ols = m12.elasticity_regression(df, "p", "q")["elasticity"]
        iv = m12.iv2sls_elasticity(df, "p", "q", "cost")
        assert iv["first_f"] > 10, iv
        assert abs(iv["elasticity"] + 0.9) < abs(ols + 0.9), (ols, iv)
    add("12 2SLS：内生数据下比 OLS 更接近真值", t12_2sls_closer_than_ols)

    # ---- 12 报童：临界比越大 Q* 越大；经验版与正态版一致 ----
    def t12_newsvendor_monotone():
        q1, r1 = m12.newsvendor_normal(100, 15, cu=2, co=1)
        q2, r2 = m12.newsvendor_normal(100, 15, cu=4, co=1)
        assert r1 < r2 and 100 < q1 < q2
        qs, _ = m12.newsvendor_quantile(np.random.default_rng(3).normal(100, 15, 5000), 2, 1)
        assert abs(qs - q1) < 3, (qs, q1)
        try:
            m12.newsvendor_quantile([1., 2., 3.], 2, 1)
        except ValueError:
            return
        raise AssertionError("小样本未抛错")
    add("12 报童：临界比单调 + 经验版≈正态版 + 小样本抛错", t12_newsvendor_monotone)

    # ---- 12 STL：结构完整、季节项可还原周期幅度 ----
    def t12_stl_shape():
        t = np.arange(70)
        s = t + 5 * np.sin(t * 2 * np.pi / 7)
        out = m12.stl_decompose(s, period=7)
        assert list(out.columns) == ["observed", "trend", "seasonal", "resid"]
        assert np.isfinite(out["seasonal"]).all() and len(out) == 70
        try:
            m12.stl_decompose(t[:10], period=7)
        except ValueError:
            return
        raise AssertionError("序列过短未抛错")
    add("12 STL：结构完整 + 短序列抛错", t12_stl_shape)

    # ---- 12 帕累托：A 类累计占比 ≤ 阈值；非法输入抛错 ----
    def t12_pareto_abc():
        df = pd.DataFrame({"item": list("abcdefghij"),
                           "sales": [90, 50, 30, 10, 8, 5, 3, 2, 1, 1]})
        out = m12.pareto_abc(df, "sales")
        assert out.iloc[0]["class"] == "A" and out.iloc[-1]["class"] == "C"
        assert out["cum_share"].iloc[-1] > 0.999
        try:
            m12.pareto_abc(pd.DataFrame({"sales": [0., 0.]}), "sales")
        except ValueError:
            return
        raise AssertionError("零合计未抛错")
    add("12 帕累托 ABC：排序标签 + 零合计抛错", t12_pareto_abc)

    # ---- 13 一致性校验：硬编码抓出、四舍五入放行、年份/编号忽略 ----
    def t13_catches_hardcoded():
        vals = [-0.27, 3.42, 8309.4]
        text = ("由回归得价格弹性为 -0.35，预测 MAPE 为 3.4%；"
                "第 3 问周利润达 8309 元（见表 2）。2023 年 6 月数据。")
        bad = m13.find_unmatched(text, vals)
        assert [v for v, _ in bad] == [-0.35], bad
        assert any("弹性" in ctx for _, ctx in bad)
    add("13 一致性校验：硬编码抓出 + 四舍五入放行 + 年份编号忽略", t13_catches_hardcoded)

    def t13_percent_and_scientific():
        vals = [0.75, 1.8e-5]
        text = "达标率为 75%，量级为 1.83e-5（见公式 3）。"  # 均在 2% 容差内，编号/日期忽略
        bad = m13.find_unmatched(text, vals)
        assert bad == [], bad
    add("13 一致性校验：百分号双口径 + 科学计数", t13_percent_and_scientific)

    return checks


def main():
    print(f"共 {len(SCRIPTS)} 个脚本，开始自检...\n")
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for script in SCRIPTS:
            name = os.path.basename(script)
            try:
                r = run([sys.executable, script], cwd=tmp, capture_output=True,
                        text=True, timeout=180,
                        env={**os.environ})
                if r.returncode == 0:
                    print(f"  [PASS] {name}")
                else:
                    print(f"  [FAIL] {name}  退出码 {r.returncode}")
                    print("  " + r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "")
                    failures.append(name)
            except Exception:
                print(f"  [FAIL] {name}  运行异常")
                traceback.print_exc()
                failures.append(name)

    print("\n函数级回归测试...\n")
    try:
        checks = regression_checks()
    except Exception:
        print("  [FAIL] 回归测试环境加载失败")
        traceback.print_exc()
        failures.append("regression_checks")
        checks = []
    import warnings
    for name, fn in checks:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fn()
            print(f"  [PASS] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}  -> {type(e).__name__}: {e}")
            failures.append(name)

    print()
    total = len(SCRIPTS) + len(checks)
    if failures:
        print(f"结果：{total - len(failures)}/{total} 通过，失败：{', '.join(failures)}")
        print("排查建议：缺依赖先 pip install -r requirements.txt；再核对上方失败项信息。")
        sys.exit(1)
    print(f"结果：{total}/{total} 全部通过，环境就绪。")


if __name__ == "__main__":
    main()
