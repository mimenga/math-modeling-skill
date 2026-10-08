# -*- coding: utf-8 -*-
"""
06 评价方法：AHP / 熵权法 / TOPSIS
用途：多指标综合评价、排序
依赖：numpy
注意：权重来源要写清依据；最后做灵敏度分析验证排序稳定性。
"""
import numpy as np

# ---------- 1. 层次分析法 AHP ----------
def ahp(matrix):
    """matrix: 成对比较矩阵（n×n），返回权重；需 CR<0.1"""
    A = np.array(matrix, dtype=float)
    n = A.shape[0]
    eigvals, eigvecs = np.linalg.eig(A)
    k = int(np.argmax(eigvals.real))
    w = np.abs(eigvecs[:, k].real)
    w = w / w.sum()                          # 归一化权重
    lamda = float(eigvals.real[k])
    if n < 3:                                # 1、2 阶矩阵恒一致，无需检验
        return w, 0.0
    CI = (lamda - n) / (n - 1)
    RI = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45]  # 1~9 阶
    ri = RI[n - 1] if n - 1 < len(RI) else 1.49
    CR = CI / ri if ri > 0 else 0.0
    return w, CR

# ---------- 2. 熵权法（客观权重） ----------
def entropy_weight(X):
    """X: 指标矩阵（行=方案，列=指标），返回各指标权重

    注意：取值完全相同的常数列零信息量，权重为 0；只有一个方案时返回等权重。
    """
    X = np.array(X, dtype=float)
    n, m = X.shape
    if n < 2:                                # 单方案无区分信息
        return np.full(m, 1.0 / m)
    rng = X.max(0) - X.min(0)
    X = (X - X.min(0)) / (rng + 1e-10)       # 归一化
    P = X / (X.sum(0) + 1e-10)
    E = -(P * np.log(P + 1e-10)).sum(0) / np.log(n)
    d = 1 - E
    d[rng <= 0] = 0.0                        # 常数列：零信息量，权重必须置 0
    if d.sum() == 0:                         # 所有列都是常数 → 等权重
        return np.full(m, 1.0 / m)
    return d / d.sum()

# ---------- 3. TOPSIS ----------
def topsis(X, w, benefit):
    """w: 权重; benefit: 各指标是否正向(1=越大越好, 0=越小越好)

    注意：w 与 benefit 必须先用 np.asarray 转换。若直接传 Python list，
    `benefit == 1` 会返回标量 False（list 与 int 比较），导致 np.where 整体
    走 else 分支，正负向指标全部失效且不报错（静默错误）。
    """
    X = np.array(X, dtype=float)
    w = np.asarray(w, dtype=float)
    benefit = np.asarray(benefit)
    norm = np.sqrt((X ** 2).sum(0))
    nz = norm > 0
    X[:, nz] = X[:, nz] / norm[nz]           # 向量归一化（全 0 列保持为 0，不产生 NaN）
    X = X * w
    best = np.where(benefit == 1, X.max(0), X.min(0))
    worst = np.where(benefit == 1, X.min(0), X.max(0))
    d_plus = np.sqrt(((X - best) ** 2).sum(1))
    d_minus = np.sqrt(((X - worst) ** 2).sum(1))
    return d_minus / (d_plus + d_minus)      # 贴近度，越大越优

if __name__ == "__main__":
    # AHP 示例（3×3 成对比较）
    A = [[1, 3, 5], [1/3, 1, 2], [1/5, 1/2, 1]]
    w, CR = ahp(A)
    print("AHP 权重:", w.round(3), "CR:", round(CR, 3))

    # 熵权法示例（第 2 列为常数 5，零信息量，权重应为 0）
    Xe = [[8, 5, 2], [6, 5, 5], [7, 5, 3]]
    print("熵权法权重(第2列常数应为0):", entropy_weight(Xe).round(3))

    # TOPSIS 示例（3 方案 3 指标，前两个正向、第三个负向）
    X = [[8, 7, 2], [6, 9, 5], [7, 6, 3]]
    score = topsis(X, w=[0.4, 0.3, 0.3], benefit=[1, 1, 0])
    print("TOPSIS 排序得分:", score.round(3))
