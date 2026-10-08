# -*- coding: utf-8 -*-
"""
05 时间序列预测
用途：趋势预测
依赖：numpy, statsmodels（ARIMA）, 指数平滑自实现
注意：数据量少（<10 点）优先灰色 GM(1,1)；有季节性先 STL 去季节（见 12 号脚本）再建模。
"""
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# ---------- 1. 指数平滑（一次/二次） ----------
def exp_smooth(y, alpha=0.3):
    """一次指数平滑。

    注意：返回的是【平滑后的拟合序列】（长度与 y 相同），不是未来预测值。
    要预测未来请直接用该平滑值作水平外推，或改用 ARIMA / GM(1,1)。
    """
    s = [y[0]]
    for t in range(1, len(y)):
        s.append(alpha * y[t] + (1 - alpha) * s[-1])
    return np.array(s)


# ---------- 2. ARIMA ----------
def arima_forecast(y, order=(1, 1, 1), steps=5):
    """返回未来 steps 期的预测值（pandas Series）。

    ARIMA 一般需要 10 个以上观测点才稳定；样本太少请改用 gm11。
    """
    model = ARIMA(np.asarray(y, dtype=float), order=order).fit()
    return model.forecast(steps=steps)

# ---------- 3. 灰色 GM(1,1)（小样本） ----------
def gm11(y, steps=1):
    """灰色 GM(1,1)，适合小样本（<10 点）。

    返回累减还原后的序列，长度为 len(y)+steps-1：
    前 len(y)-1 项是拟合值，最后 steps 项是未来预测值。
    例如 gm11(y, steps=3)[-3:] 即未来 3 期预测。

    适用性检查：GM(1,1) 要求数据为正且准指数增长；含非正值或数据近乎
    平稳（发展系数 a≈0）时直接抛错，避免静默输出无意义的预测。
    """
    y = np.array(y, dtype=float)
    if (y <= 0).any():
        raise ValueError("GM(1,1) 要求数据为正（级比检验），含非正值请改用 ARIMA 或先平移/差分")
    x1 = np.cumsum(y)                      # 一次累加
    z1 = 0.5 * (x1[1:] + x1[:-1])          # 紧邻均值
    B = np.vstack([-z1, np.ones(len(z1))]).T
    Y = y[1:]
    a, u = np.linalg.lstsq(B, Y, rcond=None)[0]   # 发展系数 a、灰作用量 u
    if abs(a) < 1e-6:
        raise ValueError("发展系数 a≈0（数据近乎平稳），GM(1,1) 不适用——"
                         "平稳序列请直接用均值或指数平滑")
    def pred(k):
        return (y[0] - u / a) * np.exp(-a * k) + u / a
    hat_x1 = np.array([pred(k) for k in range(len(y) + steps)])
    hat = np.diff(hat_x1)                  # 累减还原
    return hat

if __name__ == "__main__":
    y = [30, 35, 41, 48, 56, 65]
    print("指数平滑（拟合值，非预测）:", exp_smooth(y).round(2))
    print("GM(1,1) 下一步预测:", round(float(gm11(y, steps=1)[-1]), 3))
    print("GM(1,1) 未来 3 期:", gm11(y, steps=3)[-3:].round(2))

    # ARIMA 自检：样本量需 >=10，否则结果不稳定（此处用 12 点趋势序列）
    y2 = [30, 35, 41, 48, 56, 65, 75, 86, 98, 111, 125, 140]
    try:
        fc = arima_forecast(y2, order=(1, 1, 1), steps=3)
        print("ARIMA(1,1,1) 未来 3 期:", np.round(np.asarray(fc, dtype=float), 2))
    except Exception as e:
        print("ARIMA 未运行，请检查 statsmodels 是否安装：", e)
