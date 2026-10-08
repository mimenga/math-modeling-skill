# -*- coding: utf-8 -*-
"""
07 智能优化算法：遗传算法 / 模拟退火 / PSO
用途：大规模组合优化、非线性目标求满意解
依赖：numpy
注意：每次调用内部用 seed 新建局部 rng，保证【重复调用结果一致】。
      论文要求记录：随机种子、迭代收敛曲线、多次运行的均值与标准差。
"""
import numpy as np

RANDOM = 42


# ---------- 1. 遗传算法（求函数极小值） ----------
def genetic_algorithm(fitness, n_dim, bounds, pop=50, gen=100, pc=0.8, pm=0.1,
                      seed=RANDOM, return_history=False):
    """fitness: 目标函数; bounds: [(low, high), ...]

    return_history=True 时额外返回每代最优值数组（画收敛曲线用）
    """
    rng = np.random.default_rng(seed)
    low = np.array([b[0] for b in bounds], dtype=float)
    high = np.array([b[1] for b in bounds], dtype=float)
    population = rng.uniform(low, high, (pop, n_dim))
    history = []
    for _ in range(gen):
        fit = np.array([fitness(ind) for ind in population])
        history.append(fit.min())
        idx = fit.argsort()[:pop // 2]           # 选择最优一半
        parents = population[idx]
        offspring = []
        while len(offspring) < pop - len(parents):
            a, b = parents[rng.integers(len(parents), size=2)]
            if rng.random() < pc:                # 交叉
                c = rng.uniform(0, 1, n_dim)
                child = a * c + b * (1 - c)
            else:
                child = a.copy()
            mask = rng.random(n_dim) < pm        # 变异
            if mask.any():
                child[mask] = rng.uniform(low, high, n_dim)[mask]
            offspring.append(child)
        population = np.vstack([parents, np.array(offspring)])
    fit = np.array([fitness(ind) for ind in population])
    best_x, best_f = population[fit.argmin()], fit.min()
    if return_history:
        return best_x, best_f, np.array(history)
    return best_x, best_f


# ---------- 2. 模拟退火 ----------
def simulated_annealing(fitness, x0, bounds, T=1000, cooling=0.95, iters=200,
                        seed=RANDOM, return_history=False):
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float); best = x.copy()
    f = best_f = fitness(x)
    low = np.array([b[0] for b in bounds], dtype=float)
    high = np.array([b[1] for b in bounds], dtype=float)
    history = []
    for _ in range(iters):
        T *= cooling
        x_new = x + rng.normal(0, 0.1, len(x)) * (high - low)
        x_new = np.clip(x_new, low, high)
        f_new = fitness(x_new)
        if f_new < f or rng.random() < np.exp((f - f_new) / (T + 1e-10)):
            x, f = x_new, f_new
        if f < best_f:
            best, best_f = x.copy(), f
        history.append(best_f)
    if return_history:
        return best, best_f, np.array(history)
    return best, best_f


# ---------- 3. 粒子群 PSO ----------
def pso(fitness, n_dim, bounds, pop=30, gen=100, w=0.7, c1=1.5, c2=1.5,
        seed=RANDOM, return_history=False):
    rng = np.random.default_rng(seed)
    low = np.array([b[0] for b in bounds], dtype=float)
    high = np.array([b[1] for b in bounds], dtype=float)
    pos = rng.uniform(low, high, (pop, n_dim))
    vel = np.zeros_like(pos)
    pbest = pos.copy(); pbest_f = np.array([fitness(p) for p in pos])
    gbest = pbest[pbest_f.argmin()]; gbest_f = pbest_f.min()
    history = []
    for _ in range(gen):
        vel = (w * vel
               + c1 * rng.random((pop, n_dim)) * (pbest - pos)
               + c2 * rng.random((pop, n_dim)) * (gbest - pos))
        pos = np.clip(pos + vel, low, high)
        f = np.array([fitness(p) for p in pos])
        better = f < pbest_f
        pbest[better] = pos[better]; pbest_f[better] = f[better]
        if f.min() < gbest_f:
            gbest, gbest_f = pos[f.argmin()], f.min()
        history.append(gbest_f)
    if return_history:
        return gbest, gbest_f, np.array(history)
    return gbest, gbest_f


# ---------- 4. 多次运行统计（论文稳定性论据） ----------
def multi_run(solver, n_runs=10, seed=RANDOM, **kw):
    """用不同种子独立跑 n_runs 次，返回 (均值, 标准差, 最优解)。

    示例：
        mean, std, best_x = multi_run(pso, n_runs=10, fitness=f, n_dim=3, bounds=[(-5,5)]*3)
    """
    vals, xs = [], []
    for i in range(n_runs):
        x, v = solver(seed=seed + i, **kw)
        vals.append(v); xs.append(x)
    vals = np.asarray(vals, dtype=float)
    return vals.mean(), vals.std(), xs[int(np.argmin(vals))]


if __name__ == "__main__":
    f = lambda x: np.sum((x - 2) ** 2)          # 目标：全局最小点 (2,2,...)
    bx, bf, hist = genetic_algorithm(f, 3, [(-5, 5)] * 3, return_history=True)
    print("GA :", bx.round(4), "最优值", round(float(bf), 8), "收敛曲线长度", len(hist))
    px, pf = pso(f, 3, [(-5, 5)] * 3)
    print("PSO:", px.round(4), "最优值", round(float(pf), 8))

    # 可复现性自检：同种子两次调用必须完全一致
    a = genetic_algorithm(f, 3, [(-5, 5)] * 3)[1]
    b = genetic_algorithm(f, 3, [(-5, 5)] * 3)[1]
    print("同种子重复调用一致:", np.isclose(a, b))

    # 多次运行稳定性（论文用：均值 ± 标准差）
    mean, std, best_x = multi_run(pso, n_runs=10, fitness=f, n_dim=3, bounds=[(-5, 5)] * 3)
    print(f"PSO 10 次运行: 均值 {mean:.3e}  标准差 {std:.3e}  最优解 {best_x.round(4)}")
