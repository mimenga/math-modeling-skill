# -*- coding: utf-8 -*-
"""
02 线性与整数规划
用途：生产计划、资源分配、选址、组合优化
依赖：scipy（linprog/milp）、PuLP（可选）
注意：论文里先写清决策变量、目标函数、约束，再谈求解器。
"""
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

# ---------- 1. 线性规划：min c·x  s.t. A_ub·x <= b_ub, x>=0 ----------
def lp_solve(c, A_ub, b_ub, A_eq=None, b_eq=None):
    """c: 目标系数; A_ub/b_ub: 不等式约束; A_eq/b_eq: 等式约束

    不可行/无界时抛 ValueError（而不是静默返回 None），便于定位约束错误。
    """
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method="highs")
    if not res.success:
        raise ValueError(f"线性规划无解（status={res.status}）：{res.message}\n"
                         "排查：约束是否互相矛盾（不可行）/ 目标是否缺少上界（无界）")
    return res.x, res.fun

# ---------- 2. 整数规划（0-1 / 整数） ----------
def milp_solve(c, A_ub, b_ub, integrality, A_eq=None, b_eq=None, lb=0, ub=np.inf):
    """integrality: 每个变量是否取整(1=整数,0=连续)。A_eq/b_eq: 等式约束
    （指派/覆盖问题必用：每行每列之和=1 这类约束）。不可行时抛 ValueError。"""
    constraints = [LinearConstraint(A_ub, -np.inf, b_ub)]
    if A_eq is not None:
        constraints.append(LinearConstraint(A_eq, b_eq, b_eq))
    res = milp(c=c, constraints=constraints,
               integrality=integrality, bounds=Bounds(lb, ub))
    if not res.success:
        raise ValueError(f"整数规划无解（status={res.status}）：{res.message}\n"
                         "排查：约束是否互相矛盾（不可行）/ 目标是否缺少上界（无界）")
    return res.x, res.fun

# ---------- 3. PuLP 写法（可选依赖：pip install pulp） ----------
# 与 scipy 相比优势：变量可命名、约束可逐条命名、直接输出 LP 文件与求解日志，
# 论文需要展示标准形/约束列表时更方便。
try:
    from pulp import LpProblem, LpVariable, LpMaximize, PULP_CBC_CMD

    def pulp_demo():
        """同一示例：max 3x1+2x2 s.t. x1+x2<=4, 2x1+x2<=5, x>=0"""
        prob = LpProblem("demo", LpMaximize)
        x1 = LpVariable("x1", lowBound=0)     # lowBound=0, cat="Integer" 即整数变量
        x2 = LpVariable("x2", lowBound=0)
        prob += 3 * x1 + 2 * x2               # 目标函数（默认第一个 +=）
        prob += x1 + x2 <= 4                  # 逐条添加约束，可读性好
        prob += 2 * x1 + x2 <= 5
        prob.solve(PULP_CBC_CMD(msg=0))       # msg=1 可看求解日志
        return [x1.value(), x2.value()], prob.objective.value()

    _HAS_PULP = True
except ImportError:
    _HAS_PULP = False

if __name__ == "__main__":
    # 示例：max 3x1 + 2x2  s.t. x1 + x2 <= 4, 2x1 + x2 <= 5, x>=0
    # linprog 求 min，故取负
    c = np.array([-3, -2])
    A_ub = np.array([[1, 1], [2, 1]])
    b_ub = np.array([4, 5])
    x, val = lp_solve(c, A_ub, b_ub)
    print("线性规划最优解:", x, "最大值:", -val)

    # 整数规划示例（0-1 背包简化）
    c2 = np.array([-5, -3])          # 物品价值取负
    A2 = np.array([[2, 1]])          # 重量约束
    b2 = np.array([3])
    xi, vi = milp_solve(c2, A2, b2, integrality=np.array([1, 1]), ub=1)
    print("0-1规划最优解:", xi, "最大值:", -vi)

    if _HAS_PULP:
        xs, obj = pulp_demo()
        print("PuLP 求解结果:", xs, "最大值:", obj)
    else:
        print("未安装 PuLP（可选）：pip install pulp")
