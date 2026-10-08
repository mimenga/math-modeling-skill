# -*- coding: utf-8 -*-
"""
10_可视化模板.py
用途：论文插图统一风格。图表规范 = 编号 + 标题 + 标注 + 解读，直接决定评委好感度。
依赖：matplotlib seaborn
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 全局风格设置（统一字体、配色、清晰度）
plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei'],   # 中文字体
    'axes.unicode_minus': False,                        # 负号正常显示
    'figure.dpi': 150,
    'savefig.dpi': 300,                                 # 论文图高清
    'axes.grid': True, 'grid.alpha': 0.3,
})
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']

# ============ 1. 散点图 + 拟合线（关系/趋势） ============
def scatter_fit(x, y, xlabel='x', ylabel='y', title='', save=None):
    """x/y 可传 list 或 ndarray；x 有重复值（如重复采样）时自动按均值合并。
    去重后点数 <10 时趋势线自动降级为最小二乘直线（三次样条在小样本上会过拟合）。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=15, alpha=0.6, color=COLORS[0], label='数据点')
    # 平滑趋势线（重复 x 取均值合并）
    from scipy.interpolate import make_interp_spline
    xu, inv = np.unique(x, return_inverse=True)
    ys = np.bincount(inv, weights=y) / np.bincount(inv)
    if len(xu) < 10:                         # 小样本：直线更稳健
        k1, k0 = np.polyfit(xu, ys, 1)
        xs = np.linspace(x.min(), x.max(), 200)
        plt.plot(xs, k1 * xs + k0, '-', color=COLORS[1], lw=2, label='线性拟合')
    else:
        spl = make_interp_spline(xu, ys, k=min(3, len(xu) - 1))
        xs = np.linspace(x.min(), x.max(), 200)
        plt.plot(xs, spl(xs), '-', color=COLORS[1], lw=2, label='趋势曲线')
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.legend()
    if save: plt.savefig(save, bbox_inches='tight')
    plt.show()

# ============ 2. 折线图（时间序列/多方案对比） ============
def line_chart(data_dict, xlabel='时间', ylabel='数值', title='', x=None):
    """data_dict: {'序列名': np.array}；x 可选：各序列共用的数值型横轴坐标，
    不传则用等间隔索引（0,1,2,...）"""
    plt.figure(figsize=(9, 5))
    for i, (name, y) in enumerate(data_dict.items()):
        xs = x if x is not None else range(len(y))
        plt.plot(xs, y, 'o-', lw=2, ms=4, color=COLORS[i % len(COLORS)], label=name)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.legend()
    plt.show()

# ============ 3. 柱状图（对比/排名，支持分组与堆叠） ============
def bar_chart(labels, values, title='', ylabel='数值', highlight=None,
              stacked=False, series_names=None):
    """values: 一维序列（单系列，行为与旧版一致）或 二维结构（多系列分组/堆叠，
    形状 = (系列数, 类别数)，也可传 {'系列名': 序列} 的 dict）。
    stacked=True 时堆叠；highlight 仅单系列生效（高亮某些类别的柱）。
    """
    plt.figure(figsize=(8, 5))
    v = np.asarray(values, dtype=float) if not isinstance(values, dict) else None
    if isinstance(values, dict):
        series_names = list(values.keys())
        v = np.asarray(list(values.values()), dtype=float)
    if v.ndim == 1:                          # 单系列（兼容旧用法）
        colors_ = [COLORS[3] if (highlight and i in highlight) else COLORS[0]
                   for i in range(len(labels))]
        bars = plt.bar(labels, v, color=colors_, alpha=0.85)
        for b, val in zip(bars, v):
            plt.text(b.get_x() + b.get_width() / 2, val, f'{val:.2f}',
                     ha='center', va='bottom', fontsize=9)
    else:                                    # 多系列：分组 / 堆叠
        n_series = v.shape[0]
        xpos = np.arange(len(labels))
        if stacked:
            bottom = np.zeros(len(labels))
            for i in range(n_series):
                name = series_names[i] if series_names else f'系列{i+1}'
                plt.bar(xpos, v[i], 0.7, bottom=bottom, label=name,
                        color=COLORS[i % len(COLORS)], alpha=0.9)
                bottom += v[i]
        else:
            width = 0.8 / n_series
            for i in range(n_series):
                name = series_names[i] if series_names else f'系列{i+1}'
                plt.bar(xpos + (i - (n_series - 1) / 2) * width, v[i], width,
                        label=name, color=COLORS[i % len(COLORS)], alpha=0.9)
        plt.xticks(xpos, labels)                 # 分组/堆叠都要设，否则横轴显示 0,1,2…
        plt.legend()
    plt.ylabel(ylabel); plt.title(title)
    plt.show()

# ============ 4. 箱线图（分布对比，异常值直观） ============
def box_plot(groups, group_names=None, title='', ylabel='数值'):
    """groups: [array1, array2, ...]"""
    plt.figure(figsize=(8, 5))
    try:
        bp = plt.boxplot(groups, tick_labels=group_names, patch_artist=True,
                         widths=0.5)
    except TypeError:                        # matplotlib < 3.9 参数名是 labels
        bp = plt.boxplot(groups, labels=group_names, patch_artist=True,
                         widths=0.5)
    for patch, c in zip(bp['boxes'], COLORS):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    plt.ylabel(ylabel); plt.title(title)
    plt.show()

# ============ 5. 相关性热力图（变量关系总览） ============
def corr_heatmap(df, method='pearson', title='变量相关性热力图', save=None):
    """method: 'pearson'/'spearman'。C 题问1 常用！"""
    plt.figure(figsize=(9, 7))
    corr = df.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                vmin=-1, vmax=1, square=True,
                cbar_kws={'label': '相关系数'})
    plt.title(title)
    if save: plt.savefig(save, bbox_inches='tight')
    plt.show()
    return corr

# ============ 6. 双 y 轴（量纲不同但相关的两个量） ============
def twin_axis(x, y1, y2, label1='量1', label2='量2', title=''):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(x, y1, 'o-', color=COLORS[0], label=label1)
    ax1.set_xlabel('x'); ax1.set_ylabel(label1, color=COLORS[0])
    ax2 = ax1.twinx()
    ax2.plot(x, y2, 's--', color=COLORS[1], label=label2)
    ax2.set_ylabel(label2, color=COLORS[1])
    plt.title(title)
    plt.show()

# ============ 7. 三维曲面图（两个变量对目标的影响） ============
def surface_3d(X, Y, Z, xlabel='x', ylabel='y', zlabel='z', title='',
               cmap='viridis', cbar_label=None, save=None):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.8, edgecolor='none')
    if cbar_label is not None:               # 有色条才能从曲面读出数值
        fig.colorbar(surf, ax=ax, shrink=0.6, label=cbar_label)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
    ax.set_title(title)
    if save: plt.savefig(save, bbox_inches='tight')
    plt.show()
    return fig, ax

# ============ 8. 矩阵着色热力图（二维密度/强度分布，C 题高频） ============
def matrix_heatmap(Z, x=None, y=None, cbar_label='', cmap='viridis',
                   xlabel='', ylabel='', title='', save=None):
    """Z: 二维矩阵（行对应 y，列对应 x）。通用 pcolormesh 封装，
    与 corr_heatmap（专用于相关系数矩阵）互补。"""
    Z = np.asarray(Z, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if x is None: x = np.arange(Z.shape[1])
    if y is None: y = np.arange(Z.shape[0])
    im = ax.pcolormesh(x, y, Z, cmap=cmap, shading='auto')
    fig.colorbar(im, ax=ax, label=cbar_label or None)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if save: plt.savefig(save, bbox_inches='tight')
    plt.show()
    return fig, ax

# ============ 9. 回归残差诊断（残差图 + 正态 Q-Q 图） ============
def residual_plot(y_true, y_pred, xlabel='拟合值', ylabel='残差', title='', save=None):
    """残差-拟合值图：检查线性/方差齐性假设，残差应随机散布在 0 附近。"""
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, resid, s=45, alpha=0.75, color=COLORS[0])
    plt.axhline(0, color='#333333', lw=1.2)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    if save: plt.savefig(save, bbox_inches='tight')
    plt.show()

def qq_plot(residuals, title='', save=None):
    """正态 Q-Q 图：点贴近对角直线说明残差近似正态（t 检验/方差分析前提）。
    返回 (osm, osr, r²)。"""
    from scipy import stats
    residuals = np.asarray(residuals, dtype=float)
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist='norm')
    plt.figure(figsize=(6, 6))
    plt.scatter(osm, osr, s=35, alpha=0.75, color=COLORS[0])
    plt.plot(osm, slope * osm + intercept, '-', color=COLORS[3], lw=1.5)
    plt.xlabel('理论分位数'); plt.ylabel('样本分位数'); plt.title(title)
    plt.annotate(f'$R^2$ = {r ** 2:.4f}', xy=(0.05, 0.92), xycoords='axes fraction',
                 bbox=dict(boxstyle='round', fc='white', ec='#999999'))
    if save: plt.savefig(save, bbox_inches='tight')
    plt.show()
    return osm, osr, r ** 2

# ============ 10. 散点矩阵 EDA（多变量两两关系总览） ============
def pair_matrix(df, cols=None, save=None):
    """散点矩阵：对角线为各变量分布，其余为两两散点。cols 指定参与的列（默认全部数值列）。"""
    import seaborn as sns
    d = df[cols] if cols is not None else df.select_dtypes(include=[np.number])
    g = sns.pairplot(d, diag_kind='hist',
                     plot_kws=dict(s=8, alpha=0.45, color=COLORS[0]),
                     diag_kws=dict(color=COLORS[0], alpha=0.75))
    if save: g.figure.savefig(save, bbox_inches='tight')
    plt.show()
    return g

if __name__ == '__main__':
    np.random.seed(42)
    x = np.linspace(1, 10, 50)
    y = 0.4 * x + np.random.normal(0, 0.5, 50)
    scatter_fit(x, y, 'x', 'y', '散点+趋势线示例', '示例图1.png')
    line_chart({'方案A': np.cumsum(np.random.randn(30)) + 10,
                '方案B': np.cumsum(np.random.randn(30)) + 10}, title='折线图示例')
    bar_chart(['A', 'B', 'C', 'D'], [12.3, 8.5, 15.2, 6.1], '柱状图示例')
    box_plot([np.random.normal(0, 1, 100), np.random.normal(2, 1.5, 100)],
             ['组1', '组2'], '箱线图示例')
    import pandas as pd
    df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
    corr_heatmap(df, title='相关性热力图示例')
    print('示例完成')
