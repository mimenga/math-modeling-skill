# -*- coding: utf-8 -*-
"""
03 回归与分类
用途：预测、判别、异常分类
依赖：scikit-learn, pandas, numpy
注意：先 EDA，再特征工程；每个子问题做主模型+对比模型；固定随机种子。
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

RANDOM = 42

# ---------- 1. 回归（连续值） ----------
def regression(X, y, method="ridge"):
    """method: linear / ridge / rf"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM)
    if method == "linear":
        m = LinearRegression()
    elif method == "ridge":
        m = Ridge(alpha=1.0)
    else:
        m = RandomForestRegressor(n_estimators=100, random_state=RANDOM)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    return {
        "model": m,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
    }

# ---------- 2. 分类（0/1 或多类） ----------
def classify(X, y, method="lr"):
    """method: lr(逻辑回归) / rf(随机森林)
    注意：逻辑回归对量纲敏感，先标准化（否则乳腺癌示例会报 ConvergenceWarning）"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM, stratify=y)
    if method == "lr":
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    else:
        m = RandomForestClassifier(n_estimators=100, random_state=RANDOM)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    return {
        "model": m,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="binary" if len(set(y)) == 2 else "macro", zero_division=0),
        "Recall": recall_score(y_test, pred, average="binary" if len(set(y)) == 2 else "macro", zero_division=0),
        "F1": f1_score(y_test, pred, average="binary" if len(set(y)) == 2 else "macro", zero_division=0),
        "confusion": confusion_matrix(y_test, pred),
    }

# ---------- 3. XGBoost（可选依赖：pip install xgboost） ----------
# 表格类数据首选之一；与随机森林对比使用（多模型对比是硬加分项）。
# 注意：xgboost 1.7+ 才完整支持 sklearn 1.2+，版本不匹配时可能报错。
try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

def xgb_classify(X, y):
    """XGBoost 分类（与 classify() 同口径的评估指标，便于论文对比表格）"""
    if not _HAS_XGB:
        raise RuntimeError("未安装 xgboost：pip install xgboost")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM, stratify=y)
    m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                      eval_metric="logloss", random_state=RANDOM)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    return {
        "model": m,
        "Accuracy": accuracy_score(y_test, pred),
        "F1": f1_score(y_test, pred, average="binary" if len(set(y)) == 2 else "macro", zero_division=0),
        "confusion": confusion_matrix(y_test, pred),
    }

# ---------- 4. 特征重要性 ----------
def feature_importance(X, y, names=None):
    """names 可传 list 或 ndarray（sklearn 的 data.feature_names 就是 ndarray）。

    y 为连续值时自动改用 RandomForestRegressor：分类器把连续 y 当类别，
    会静默给出无意义的重要性。
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.number) and np.unique(y).size > 20:   # 连续目标 → 回归（整数编码的目标也算）
        rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM)
    else:
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_,
                    index=names if names is not None else range(X.shape[1]))
    return imp.sort_values(ascending=False)

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X, y = data.data, data.target
    print("逻辑回归:", classify(X, y, "lr"))
    print("随机森林:", classify(X, y, "rf"))
    if _HAS_XGB:
        print("XGBoost:", xgb_classify(X, y))
    else:
        print("未安装 xgboost（可选）：pip install xgboost")
