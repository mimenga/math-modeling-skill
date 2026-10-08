# 数模合作 skill · 安装与使用说明

这是一套数模竞赛 AI 协作 skill + 知识库（含 2025C NIPT 专题、建模方法库、论文写作规范与代码模板）。

## 一、安装（三步）

### 1. 复制 skill

把本仓库内容放入用户级 skill 目录下的 `math-modeling-competition` 文件夹（克隆后把仓库文件夹重命名为该名称即可）：

```
ZCode:    C:\Users\你的用户名\.zcode\skills\math-modeling-competition\
WorkBuddy: C:\Users\你的用户名\.workbuddy\skills\math-modeling-competition\
最终应存在: ...\math-modeling-competition\SKILL.md
```

（如果你用的不是以上工具：`references/` 下的 61 张知识卡片是纯 Markdown，任何 AI 助手都能读——把卡片喂给你用的 AI 即可。）

### 2. 装 Python 依赖并自检

```bash
cd math-modeling-competition/scripts
pip install -r requirements.txt
python smoke_test.py
```

看到 `40/40 全部通过`（13 个脚本 + 27 项函数级回归测试）即环境就绪（约 2 分钟）。

### 3. 关于论文全文等外部资料

他人获奖论文全文、培训课件等第三方资料因版权原因**不随仓库分发**。需要时可从竞赛官网（www.mcm.edu.cn）等公开渠道自行获取，再按 `references/08_协作与进度/02_知识库与蒸馏协议.md` 的流程蒸馏入库。

其余功能（建模方法库、NIPT 四问解法、论文写作规范与句式/灵敏度模板、13 个代码模板、论文模板）均开箱即用。

## 二、怎么用（赛前 vs 赛中）

| 场景 | 入口 |
|---|---|
| 赛前通读 | `references/01_参赛总纲/05_参赛总览_先读我.md` |
| 选模型 | `references/02_建模方法库/00_方法选择决策树.md` |
| C 题打法 | `references/06_NIPT_C题专题/`（评阅要点 + 四问解法路线） |
| 写摘要 | `references/04_论文写作/02_摘要写作指南.md` |
| 套代码 | `scripts/`（先跑 `09_数据处理与EDA.py`，模型对比用 `11_科研绘图模板.py`） |
| 套论文模板 | `templates/`（Word 版直接改，LaTeX 版 xelatex 编译两遍） |

## 三、重要合规提醒（2026 AI 新规）

- **核心建模必须由队员主导**，AI 生成内容未经人工审查直接提交 = 取消评奖资格；
- 论文参考文献前必须加 **AI 工具使用声明**，用 AI 的队还要在支撑材料附 `AI工具使用详情.pdf`；
- 详见 `references/01_参赛总纲/01_参赛关键信息与AI新规.md`（含声明模板）。

## 四、三人分工建议

见 `references/01_参赛总纲/03_三人分工与AI协作.md`——建议一人建模主导、一人编程、一人写作，AI 定位"超级工具人"：查资料、写代码初稿、润色、查逻辑。
