# -*- coding: utf-8 -*-
"""
13 数字一致性校验（论文内容硬伤的最后一道防线）
用途：论文文本中手写的数值必须能在结果文件（results.json / 结果表 csv）中找到
     对应值；找不到 = 疑似"凭记忆硬编码"错误，逐条列出供人工核对。
     ——背景：历史测试中论文曾 3 处正文数字与图表结果矛盾（如正文写 0.75、
     相关矩阵实为 0.66），评委随手一对即穿帮，全靠本类校验提前拦下。
运行：python 13_数字一致性校验.py 论文文本.txt results.json
     python 13_数字一致性校验.py 论文文本.csv 结果表.csv --rel 0.02
     无参数运行 = 内置示例自检（退出码 0）。
     txt 可由 docx 另存/用 pandoc 导出；也可 import 后对字符串直接调 find_unmatched()。
匹配规则：论文数字 n 通过，若结果值中存在 r 满足 |n-r| <= max(rel*|r|, abs_floor)
     ——默认 rel=0.02（2%），覆盖四舍五入到 2~3 位有效数字的常见写法；
     百分号数字同时按原值与 /100 两口径匹配（75% ↔ 0.75）；
     年份、日期（X 年/月/日）、图表/式/问/第 X 编号自动忽略；白名单可再放行页码等。
退出码：发现疑点 = 1（可直接接进提交流程当闸门），干净 = 0。
"""
import json
import re
import sys

REL_TOL = 0.02
ABS_FLOOR = 1e-9

# 数字：含千分位逗号、小数、负号、科学计数（1.2e-3）
_NUM_RE = re.compile(r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+|-?\d+)"
                     r"(?:[eE]([+-]?\d+))?")
# 前缀为 图/表/式/问/附/章节/第 的编号、日期后缀、独立年份 → 忽略
_IGNORE_PREFIX = ("图", "表", "式", "问", "附", "节", "章", "第")
_DATE_SUFFIX = ("年", "月", "日", "时", "分")


def extract_numbers(text):
    """从文本提取 (候选值集合, 起点, 终点) 列表，供比对与定位。"""
    out = []
    for m in _NUM_RE.finditer(text):
        raw = m.group(1).replace(",", "")
        val = float(raw)
        if m.group(2):                                   # 科学计数
            val *= 10 ** int(m.group(2))
        prefix = text[max(0, m.start() - 2):m.start()].strip()
        if prefix[-1:] in _IGNORE_PREFIX:
            continue                                     # 图/表/第 X 类编号
        after = text[m.end():m.end() + 2].lstrip()[:1]   # 允许"6 月"式空格
        if after in _DATE_SUFFIX:
            continue                                     # 2023 年 / 6 月 / 5 日
        if m.group(2) is None and "." not in raw and 1900 <= val <= 2100:
            continue                                     # 独立年份
        cands = {val}
        if after == "%":
            cands.add(val / 100)                         # 75% ↔ 0.75
        out.append((cands, m.start(), m.end()))
    return out


def load_result_values(path):
    """从 json（递归取所有数值）/ csv / txt（每行一数）读取结果值集合。"""
    vals = []

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            vals.append(float(o))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            walk(json.load(f))
    elif path.endswith(".csv"):
        import csv
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                for cell in row:
                    try:
                        vals.append(float(cell.replace(",", "")))
                    except ValueError:
                        pass
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    vals.append(float(line.strip().replace(",", "")))
                except ValueError:
                    pass
    if not vals:
        raise ValueError(f"{path} 中未找到任何数值")
    return vals


def find_unmatched(paper_text, result_values, rel_tol=REL_TOL, ignore_numbers=()):
    """返回论文中找不到结果支撑的数字 [(值, 上下文), ...]。"""
    res = list(result_values) + list(ignore_numbers)

    def matched(c):
        return any(abs(c - r) <= max(rel_tol * abs(r), ABS_FLOOR) for r in res)

    bad = []
    for cands, s, e in extract_numbers(paper_text):
        if not any(matched(c) for c in cands):
            ctx = paper_text[max(0, s - 25):e + 15].replace("\n", " ")
            bad.append((next(iter(cands)), ctx.strip()))
    return bad


def main(argv):
    if len(argv) >= 3:
        with open(argv[1], encoding="utf-8") as f:
            text = f.read()
        vals = load_result_values(argv[2])
        bad = find_unmatched(text, vals)
        print(f"结果文件含 {len(vals)} 个数值；论文数字比对完成。")
        if bad:
            print(f"发现 {len(bad)} 个疑点数字（结果文件中找不到支撑）：")
            for v, ctx in bad:
                print(f"  ✗ {v:g}  …{ctx}…")
            print("请逐条核对：是手写笔误就改成结果值；确属另一来源则记入 ignore。")
            return 1
        print("✓ 全部数字可在结果文件中找到对应，通过。")
        return 0

    # ---- 内置示例自检：利润/MAPE 应通过，硬编码的弹性应被抓出 ----
    results = {"q2": {"elasticity": -0.27, "mape": 3.42}, "q3": {"week_profit": 8309.4}}
    vals = []
    for v in results.values():
        vals.extend(v.values())
    text = ("由回归得价格弹性为 -0.35，预测 MAPE 为 3.4%；"
            "第 3 问周利润达 8309 元（见表 2）。2023 年 6 月数据。")
    bad = find_unmatched(text, vals)
    caught = any(abs(v + 0.35) < 1e-9 for v, _ in bad)
    clean = not any(abs(v - 8309) < 1 or abs(v - 3.4) < 0.1 for v, _ in bad)
    print("示例：正文 {8309 元, 3.4%, -0.35} vs 结果 {8309.4, 3.42, -0.27}")
    for v, ctx in bad:
        print(f"  ✗ 抓到疑点 {v:g}  …{ctx}…")
    assert caught and clean, "示例校验逻辑失效"
    print("✓ 自检通过：四舍五入放行、硬编码抓出、年份/表号忽略。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
