# 现成代码模板库

赛前把这里每个脚本跑通一遍，比赛时按题改参数即可。统一约定：**所有随机过程固定随机种子**，保证结果可复现（论文要求）。

## 环境自检（换电脑/新环境必做）

```bash
pip install -r requirements.txt
python smoke_test.py
```

`smoke_test.py` 会以 headless 模式逐个运行全部模板并报告 PASS/FAIL，全部通过即环境就绪。可选依赖（pulp / xgboost）未安装不算失败，对应演示自动跳过；`matlab/` 下的 4 个脚本需在 MATLAB 中单独验证。

## 依赖

> 赛前务必把下面每个脚本跑通一遍：08 图论需要 networkx，09/10 需要 seaborn，漏装会直接 ModuleNotFoundError。
> 本机 pandas 若为 3.x，注意 `fillna(method=...)` 已移除，一律改用 `ffill()` / `bfill()`。

```bash
pip install -r requirements.txt
# 等价于：
pip install numpy pandas scipy scikit-learn matplotlib statsmodels seaborn networkx
# 可选：PuLP（整数规划）、xgboost
pip install pulp xgboost
```

| 脚本 | 额外依赖 |
|---|---|
| `08_图论.py` | networkx |
| `09_数据处理与EDA.py`、`10_可视化模板.py` | seaborn |
| `05_时间序列.py` | statsmodels |

## 模板清单

| 文件 | 用途 | 常用场景 |
|------|------|----------|
| `01_插值与拟合.py` | 插值、多项式/样条拟合、最小二乘 | 数据补点、曲线拟合、参数估计 |
| `02_线性与整数规划.py` | LP / 整数规划 / 0-1（含等式约束）/ PuLP 写法（可选）；约束非线性用 `matlab/非线性优化_fmincon.m` | 生产计划、资源分配、选址 |
| `03_回归与分类.py` | 线性/岭/逻辑回归、随机森林、XGBoost（可选） | 预测、判别、异常分类 |
| `04_聚类.py` | K-means、层次、DBSCAN | 分群（注意：分组若有优化目标勿直接聚类） |
| `05_时间序列.py` | 指数平滑、ARIMA、灰色 GM(1,1) | 趋势预测 |
| `06_评价方法.py` | AHP、熵权法、TOPSIS | 综合评价、排序 |
| `07_智能优化.py` | 遗传算法、模拟退火、PSO | 大规模组合优化 |
| `08_图论.py` | 最短路、最小生成树、最大流 | 网络、路径、运输 |
| `09_数据处理与EDA.py` | 缺失值、异常值(3σ/箱线图)、标准化、哑变量、相关性热力图 | **所有数据题第一步**（建议最先跑） |
| `10_可视化模板.py` | matplotlib 散点/折线/柱状/箱线/热力图统一风格 | 论文插图 |
| `11_科研绘图模板.py` | 泰勒图 / 配对雨云图 / SHAP 蜂群图（期刊级，蒸馏自 1start 套件） | 模型评价与对比、特征影响分析 |
| `12_C题定价补货专题.py` | log-log 弹性回归 / 内生性 2SLS / 分位数回归 / 报童补货（正态+经验版）/ STL 分解 / 帕累托 ABC，DataFrame 进、dict/DataFrame 出（可写 results.json） | C 题定价-预测-补货全链路（对照 `02/06`、`03/06` 卡） |
| `13_数字一致性校验.py` | 论文文本数字 ↔ 结果文件比对，抓出"凭记忆硬编码"的矛盾数字 | **论文定稿前必跑**（SKILL.md 硬约束第 9 条） |

### MATLAB 补充（`matlab/`）

| 文件 | 用途 |
|------|------|
| `规划求解_linprog_intlinprog.m` | 线性/整数规划 |
| `非线性优化_fmincon.m` | 约束非线性规划 |
| `拟合与插值_polyfit_interp.m` | 多项式拟合、插值 |
| `启发式_遗传算法示例.m` | 遗传算法工具箱示例 |

## 使用约定

- 每个脚本可独立运行，头部注释写清输入输出。
- 跑完记录：随机种子、关键超参数、评价指标，作为论文素材。
- 图表统一存 `figures/`，命名与论文图号对应（如 `fig3_roc_curve.png`）。
