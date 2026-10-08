# -*- coding: utf-8 -*-
"""
11 科研绘图模板：泰勒图 / 配对雨云图 / SHAP 蜂群图
用途：论文中"模型评价与对比"环节的期刊级配图（蒸馏自 1start mathmodel figure-templates 并通用化改造）
依赖：numpy matplotlib
注意：
- 三个函数都是通用 API，传入自己的数据即可；直接运行脚本用构造数据出演示图；
- 期刊惯例：图内不写大标题，标题交给论文 caption；统一输出 300dpi PNG + 矢量 PDF 到 figures/；
- SHAP 蜂群图为手绘实现，无需安装 shap 库；真实 SHAP 值可由 XGBoost(model.predict(..., pred_contribs=True))
  等方式计算后传入（最后一步是 bias 列，需去掉）。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.sans-serif": ["SimHei", "Microsoft YaHei"],  # 中文字体（与 10_可视化模板 一致）
    "axes.unicode_minus": False,                        # 负号显示
})

RANDOM = 42  # 全库统一随机种子


def save_fig(fig, name, dpi=300):
    """保存到 figures/name.png + name.pdf（矢量，论文插入用）"""
    import os
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: figures/{name}.png / .pdf")


# ============ 1. 泰勒图：多模型综合对比（标准差 + 相关系数 + RMS） ============
def taylor_diagram(models, ref_std=1.0, rmax=None, output=None):
    """泰勒图：一张图同时比较多个模型的 std / 相关系数 / 中心化 RMS。

    models: [(名称, 标准差, 相关系数), ...]，如 [("XGBoost", 0.98, 0.975), ("LMM", 1.05, 0.960)]
    ref_std: 观测值（参照）标准差，图中自动标 "Observed" 点
    rmax:   半径上限，默认取 max(std)*1.25

    用法：论文中"多种模型预测精度对比"一节；比单独报 R² 表格更有说服力。
    """
    stds = [m[1] for m in models] + [ref_std]
    rmax = rmax or max(stds) * 1.25
    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    theta = np.linspace(0, np.pi / 2, 300)
    # 标准差弧线网格
    for radius in np.arange(0.25, rmax, 0.25):
        if radius < max(stds) * 1.2:
            ax.plot(radius * np.cos(theta), radius * np.sin(theta), color="#cfcfcf", lw=0.5, zorder=0)
    # 相关系数射线
    corr_ticks = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    for corr in corr_ticks:
        t = np.arccos(corr)
        ax.plot([0, rmax * np.cos(t)], [0, rmax * np.sin(t)], color="#dedede", lw=0.5, zorder=0)
        ax.text(rmax * 1.03 * np.cos(t), rmax * 1.03 * np.sin(t), f"{corr:g}",
                fontsize=7, ha="center", va="center",
                rotation=np.degrees(t) - 90, rotation_mode="anchor")
    # 中心化 RMS 等值线（围绕 Observed 点的虚线圆弧）
    phi = np.linspace(0, np.pi, 400)
    for rms in [0.25, 0.5, 0.75, 1.0]:
        x = ref_std + rms * np.cos(phi)
        y = rms * np.sin(phi)
        mask = (x >= 0) & (y >= 0) & (x**2 + y**2 <= rmax**2)
        ax.plot(x[mask], y[mask], ls="--", color="#bdbdbd", lw=0.5, zorder=0)
    # 参照标准差弧线 + 外框
    ax.plot(ref_std * np.cos(theta), ref_std * np.sin(theta), ls="--", color="#cc7c8f", lw=0.8)
    ax.plot(rmax * np.cos(theta), rmax * np.sin(theta), color="#999999", lw=0.8)

    # Observed 参照点
    ax.scatter(ref_std, 0, s=30, c="black", zorder=5)
    ax.text(ref_std, -0.03 * rmax, "Observed", fontsize=8, ha="center", va="top")
    # 模型点
    for name, std, corr in models:
        t = np.arccos(np.clip(corr, 0, 1))
        ax.scatter(std * np.cos(t), std * np.sin(t), s=28, zorder=5, label=name)

    ax.set_xlim(0, rmax); ax.set_ylim(0, rmax)
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Standard Deviation", fontsize=9)
    ax.set_ylabel("Standard Deviation", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    if output:
        save_fig(fig, output)
    return fig, ax


# ============ 2. 配对雨云图：处理前后/两组对比（半小提琴+散点+箱线+均值连线） ============
def _kde_1d(values, grid, bw_adjust=1.0):
    """高斯核密度（归一化到峰值 1），仅用 numpy 实现，避免依赖 scipy/seaborn"""
    std = max(np.std(values, ddof=1), 1e-3)
    bw = max(1.06 * std * len(values) ** (-1 / 5) * bw_adjust, 1e-6)
    z = (grid[:, None] - values[None, :]) / bw
    dens = np.exp(-0.5 * z**2).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
    return dens / dens.max()


def paired_raincloud(pre, post, names=("Pre", "Post"), ylabel="", colors=("#d7191c", "#2c7bb6"), output=None):
    """配对雨云图：同一样本在两个条件（如处理前/后、优化前/后方案）下的分布对比。

    pre / post: 两组等长（配对）或不等长数值数组
    一张图同时呈现：分布形状（半小提琴）+ 个体离散（散点）+ 四分位（箱线）+ 均值变化趋势（连线）

    用法：论文中"方案改进效果""干预前后指标变化"等对比场景，比双箱线图信息量大一档。
    """
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    rng = np.random.default_rng(RANDOM)
    groups = [np.asarray(pre, float), np.asarray(post, float)]
    all_v = np.concatenate(groups)
    lo, hi = all_v.min() - 0.15 * np.ptp(all_v), all_v.max() + 0.15 * np.ptp(all_v)
    grid = np.linspace(lo, hi, 240)

    for i, (vals, name, color) in enumerate(zip(groups, names, colors)):
        x0 = 0.32 + i * 1.0  # 组中心
        dens = _kde_1d(vals, grid) * 0.30
        # 半小提琴：Pre 朝左、Post 朝右，中间对称留白
        if i == 0:
            ax.fill_betweenx(grid, x0 - dens, x0, facecolor=color, alpha=0.55, lw=0)
        else:
            ax.fill_betweenx(grid, x0, x0 + dens, facecolor=color, alpha=0.55, lw=0)
        # 散点（横向抖动）
        jitter = rng.normal(0, 0.025, len(vals))
        ax.scatter(x0 + 0.18 + jitter * 6, vals, s=14, c=color, alpha=0.55, edgecolors="none", zorder=4)
        # 箱线（窄、无异常点）
        bp = ax.boxplot(vals, positions=[x0 + 0.34], widths=0.07, patch_artist=True,
                        showfliers=False, whis=(0, 100), zorder=5)
        for b in bp["boxes"]:
            b.set(facecolor=color, alpha=0.7)
        for k in ("whiskers", "caps", "medians"):
            for item in bp[k]:
                item.set(color="#333333", lw=1.0)

    # 均值连线（配对变化趋势一眼可见）
    means = [g.mean() for g in groups]
    ax.plot([0.66, 1.66], means, color="#333333", lw=1.2, ls="-", marker="D", ms=5, zorder=6)

    ax.set_xlim(-0.15, 2.35)
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    if output:
        save_fig(fig, output)
    return fig, ax


# ============ 3. SHAP 蜂群图：特征重要性 + 方向性（手绘实现，无需 shap 库） ============
def shap_beeswarm(feature_names, shap_values, feature_values=None, output=None, max_display=12):
    """SHAP 蜂群图：每个点=一个样本，横轴 SHAP 值（对预测的贡献），颜色=特征值高低。

    feature_names: 特征名列表（len = K）
    shap_values:   (N, K) SHAP 矩阵，XGBoost: model.get_booster().predict(DMatrix, pred_contribs=True)[:, :-1]
    feature_values:(N, K) 原始特征值矩阵（用于着色）；None 则不按值着色
    max_display:   最多显示前若干个重要特征（按 mean|SHAP| 排序）

    用法：机器学习类模型的"特征影响分析"；比单纯特征重要性条形图多了方向信息
    （正值推向预测增大、负值推向减小），评委认可度高。
    """
    sv = np.asarray(shap_values, float)
    n_feat = sv.shape[1]
    # 按 mean|SHAP| 降序排列特征
    order = np.argsort(np.abs(sv).mean(axis=0))[::-1][:max_display][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(order) + 1.6))
    rng = np.random.default_rng(RANDOM)
    cmap = plt.cm.coolwarm
    vmin = vmax = None
    if feature_values is not None:
        fv = np.asarray(feature_values, float)[:, order]
        vmin, vmax = np.nanpercentile(fv, 2), np.nanpercentile(fv, 98)

    for row, fi in enumerate(order):
        vals = sv[:, fi]
        x_range = max(np.abs(vals).max(), 1e-6)
        # 蜂群排布：SHAP 越接近 0 抖动越大（数据密），越极端越贴合中线——形成蜂群形态
        dens = np.exp(-0.5 * (vals / (x_range * 0.35)) ** 2)
        spread = 0.10 + 0.22 * dens
        y = row + rng.normal(scale=spread, size=len(vals))
        if feature_values is not None:
            colors = np.clip((sv[:, fi] * 0 + fv[:, row] - vmin) / (vmax - vmin + 1e-9), 0, 1)
            ax.scatter(vals, y, s=7, c=cmap(colors), alpha=0.75, edgecolors="none", zorder=3)
        else:
            ax.scatter(vals, y, s=7, c="#534AB7", alpha=0.7, edgecolors="none", zorder=3)

    ax.axvline(0, color="#444444", lw=0.9, alpha=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=10)
    ax.set_xlabel("SHAP value (对预测结果的贡献)", fontsize=10)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.spines[["top", "right"]].set_visible(False)
    # 颜色条
    if feature_values is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
        cb = fig.colorbar(sm, ax=ax, pad=0.02, aspect=12)
        cb.set_label("Feature value (低 → 高)", fontsize=9)
    if output:
        save_fig(fig, output)
    return fig, ax


if __name__ == "__main__":
    # ---- 演示 1：泰勒图（构造 5 个模型的 std/corr） ----
    taylor_diagram(
        [("XGBoost", 0.98, 0.975), ("LMM", 1.05, 0.960), ("OLS", 0.82, 0.905),
         ("RandomForest", 1.10, 0.940), ("GPR", 1.02, 0.968)],
        ref_std=1.0, output="demo_1_taylor")

    # ---- 演示 2：配对雨云图（构造优化前/后两组配对数据） ----
    rng = np.random.default_rng(RANDOM)
    n = 60
    pre = rng.normal(3.0, 0.4, n)
    post = pre + rng.normal(0.45, 0.25, n)   # 处理后系统性提升
    paired_raincloud(pre, post, names=("优化前", "优化后"), ylabel="指标值",
                     output="demo_2_raincloud")

    # ---- 演示 3：SHAP 蜂群图（构造 8 特征模拟数据） ----
    feat_names = ["孕周GA", "BMI", "年龄", "身高", "体重", "孕次", "采样孕周", "读段数"]
    N = 200
    fv = rng.normal(0, 1, (N, 8))
    shap_sim = fv * np.array([1.6, 1.1, 0.5, 0.25, 0.9, 0.15, 0.3, 0.05]) + rng.normal(0, 0.1, (N, 8))
    shap_beeswarm(feat_names, shap_sim, feature_values=fv, output="demo_3_shap")

    print("示例完成：三张演示图已保存到 figures/")
