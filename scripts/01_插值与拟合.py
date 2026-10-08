# -*- coding: utf-8 -*-
"""
01 插值与拟合
用途：数据补点、曲线拟合、参数估计
依赖：numpy, scipy, matplotlib
输入：x, y 数据点；输出：插值曲线 / 拟合曲线 + 系数
"""
import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]  # 中文显示（带回退）
plt.rcParams["axes.unicode_minus"] = False

# ---------- 1. 插值（过点补值） ----------
def interp_demo(x, y, x_new):
    """三种插值：线性 / 三次样条 / 拉格朗日。

    x 有重复值时自动按均值合并（真实采样数据常见，样条要求 x 严格递增）。
    拉格朗日插值用 BarycentricInterpolator 实现：scipy.interpolate.lagrange
    已弃用，SciPy 1.20 将移除。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xu, inv = np.unique(x, return_inverse=True)
    yu = np.bincount(inv, weights=y) / np.bincount(inv)
    f_linear = interpolate.interp1d(xu, yu, kind="linear")
    f_cubic = interpolate.CubicSpline(xu, yu)
    f_lagrange = interpolate.BarycentricInterpolator(xu, yu)
    return f_linear(x_new), f_cubic(x_new), f_lagrange(x_new)

# ---------- 2. 多项式拟合（最小二乘，带噪声） ----------
def poly_fit(x, y, deg=2):
    """返回多项式系数（高次在前）"""
    return np.polyfit(x, y, deg)

# ---------- 3. 自定义函数拟合（非线性最小二乘） ----------
def custom_fit(func, x, y, p0):
    """func 是自定义模型函数，p0 是初值"""
    popt, pcov = curve_fit(func, x, y, p0=p0)
    return popt, pcov

if __name__ == "__main__":
    # 示例：带噪声的二次曲线拟合
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2 * x ** 2 - 3 * x + 5 + np.random.normal(0, 2, x.size)

    coef = poly_fit(x, y, deg=2)
    y_fit = np.polyval(coef, x)

    plt.scatter(x, y, label="原始数据", s=20)
    plt.plot(x, y_fit, "r-", label=f"拟合: {coef[0]:.2f}x²+{coef[1]:.2f}x+{coef[2]:.2f}")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend()
    plt.title("多项式拟合示例")
    plt.show()
