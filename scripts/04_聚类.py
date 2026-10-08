# -*- coding: utf-8 -*-
"""
04 聚类
用途：无监督分群
依赖：scikit-learn
注意：若分组带有优化目标/协变量（如 NIPT 按 BMI 分组），勿直接聚类，
     应改用"差异性指标切分"，见 references/06_NIPT_C题专题。
"""
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score

RANDOM = 42

def kmeans(X, n_clusters=3):
    m = KMeans(n_clusters=n_clusters, random_state=RANDOM, n_init=10)
    labels = m.fit_predict(X)
    return labels, m.cluster_centers_

def hierarchy(X, n_clusters=3):
    return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X)

def dbscan(X, eps=0.5, min_samples=5):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

def evaluate(X, labels):
    """轮廓系数 + CH 指数（越大越好，标签需 ≥2 类）。

    DBSCAN 的 -1 噪声点不计入评估（否则会把噪声当簇，拉低轮廓系数）。
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    mask = labels != -1                      # 排除 DBSCAN 噪声点
    if len(set(labels[mask])) < 2:
        return None, None
    return silhouette_score(X[mask], labels[mask]), calinski_harabasz_score(X[mask], labels[mask])

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=3, random_state=RANDOM)
    labels, _ = kmeans(X, 3)
    sil, ch = evaluate(X, labels)
    print(f"轮廓系数={sil:.3f}, CH指数={ch:.1f}")
