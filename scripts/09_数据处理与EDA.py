# -*- coding: utf-8 -*-
"""
09 数据处理与EDA.py
用途：所有数据题的第一步——读数据、摸数据、清洗、特征工程。
依赖：pandas numpy matplotlib seaborn
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False                       # 负号正常显示

# ============ 1. 读入数据（按需选择） ============
# CSV:  df = pd.read_csv('data.csv', encoding='gbk')   # 中文数据常用 gbk
# Excel: df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
# SPSS: 需先另存为 csv
df = pd.DataFrame()  # 示例占位，实际替换为你的数据

# ============ 2. 数据概况（必做） ============
def overview(df):
    print('形状:', df.shape)          # (行数, 列数)
    print('列名:', list(df.columns))
    print('数据类型:\n', df.dtypes)
    print('缺失值统计:\n', df.isnull().sum())
    print('描述统计:\n', df.describe().T)   # 均值/标准差/分位数
    return df

# ============ 3. 缺失值处理 ============
def handle_missing(df, strategy='median'):
    """strategy: 'drop'删除 / 'mean'均值 / 'median'中位数 / 'ffill'前向填充

    返回新 DataFrame，不修改传入的 df（与 normalize/derive_features 行为一致）。
    """
    df = df.copy()
    if strategy == 'drop':
        return df.dropna()
    elif strategy in ('mean', 'median'):
        num_cols = df.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            if df[c].isnull().any():
                fill_val = df[c].mean() if strategy == 'mean' else df[c].median()
                df[c] = df[c].fillna(fill_val)
    elif strategy == 'ffill':
        # 注意：pandas 2.1+ 已弃用、pandas 3.0 已移除 fillna(method='ffill')，必须用 ffill()
        return df.ffill()
    elif strategy == 'bfill':
        return df.bfill()
    return df

# ============ 4. 异常值检测 ============
def detect_outliers_3sigma(series):
    """3σ 原则：|x-mean| > 3*std 视为异常"""
    mu, std = series.mean(), series.std()
    return series[(series - mu).abs() > 3 * std].index

def detect_outliers_iqr(series):
    """箱线图 IQR 法：超出 [Q1-1.5IQR, Q3+1.5IQR] 视为异常"""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)].index

# ============ 5. 标准化 / 归一化 ============
def normalize(df, cols=None, method='zscore'):
    """method: 'zscore' 或 'minmax'。距离类算法(SVM/kNN/K-means)必须做！"""
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns
    for c in cols:
        if method == 'zscore':
            s = df[c].std()
            if s < 1e-12:                        # 常数列防除零（静默产出 NaN 列）
                df[c] = 0.0
                continue
            df[c] = (df[c] - df[c].mean()) / s
        else:  # minmax
            df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min() + 1e-9)
    return df

# ============ 6. 哑变量（定类变量 → 0/1） ============
def make_dummies(df, col):
    """性别/地区/等级等定性变量转哑变量（k 类生成 k-1 或 k 列）"""
    return pd.get_dummies(df, columns=[col], drop_first=True)

# ============ 7. 衍生变量示例（BMI 式组合特征） ============
def derive_features(df):
    """示例：由身高体重生成 BMI；可按题目背景自行扩展"""
    df = df.copy()
    if '身高' in df and '体重' in df:
        df['BMI'] = df['体重'] / (df['身高'] / 100) ** 2  # 身高cm转m
    return df

# ============ 8. 可视化速览（论文素材） ============
def quick_plots(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # 数值列分布
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) >= 1:
        df[num_cols[0]].hist(bins=30, ax=axes[0, 0])
        axes[0, 0].set_title(f'{num_cols[0]} 分布')
    # 箱线图（异常值直观）
    if len(num_cols) >= 2:
        df[num_cols[:2]].boxplot(ax=axes[0, 1])
        axes[0, 1].set_title('箱线图')
    # 相关性热力图
    if len(num_cols) >= 2:
        sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f',
                    cmap='RdBu_r', ax=axes[1, 0])
        axes[1, 0].set_title('相关性热力图')
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/数据概览.png', dpi=300)  # 统一存 figures/，≥300 dpi
    plt.show()

if __name__ == '__main__':
    # 示例：构造一份假数据演示
    np.random.seed(42)
    df = pd.DataFrame({
        '年龄': np.random.randint(20, 40, 200),
        '身高': np.random.normal(165, 8, 200),
        '体重': np.random.normal(60, 10, 200),
        '性别': np.random.choice(['男', '女'], 200),
    })
    df.loc[5, '体重'] = np.nan          # 造一个缺失
    df.loc[10, '年龄'] = 999            # 造一个异常
    overview(df)
    df = handle_missing(df, 'median')
    df = derive_features(df)
    print('异常年龄索引:', detect_outliers_3sigma(df['年龄']).tolist())
    quick_plots(df)
    print('示例完成')
