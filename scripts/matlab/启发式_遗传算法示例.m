%% ============================================================
% 启发式_遗传算法示例.m
% 用途：大规模/非线性优化问题的兜底方案（MATLAB 全局优化工具箱）
% 例：min f = sum(x.^2 - 10*cos(2*pi*x) + 10)  （Rastrigin 多峰函数）
% 结果：理论最优 0，位于 x = [0 0 0 0 0]
% ============================================================
clear; clc;

%% 1. 用 ga 函数（Global Optimization Toolbox）
rng(42);                              % 固定种子，可复现
fun = @(x) sum(x.^2 - 10*cos(2*pi*x) + 10);
nvars = 5;
lb = -5.12 * ones(1, 5);
ub = 5.12 * ones(1, 5);
options = optimoptions('ga', 'Display', 'iter', 'MaxGenerations', 200, ...
    'PopulationSize', 50, 'PlotFcn', @gaplotbestf);
[x, fval] = ga(fun, nvars, [], [], [], [], lb, ub, [], options);
fprintf('GA 最优解: %s\n', mat2str(x, 4));
fprintf('GA 最优值: %.6f\n', fval);

%% 2. 模拟退火 simulannealbnd
options_sa = optimoptions('simulannealbnd', 'Display', 'final', ...
    'MaxIterations', 300);
[x_sa, fval_sa] = simulannealbnd(fun, zeros(1, 5), lb, ub, options_sa);
fprintf('SA 最优解: %s\n', mat2str(x_sa, 4));
fprintf('SA 最优值: %.6f\n', fval_sa);

%% 3. 粒子群 particleswarm
options_ps = optimoptions('particleswarm', 'Display', 'final', ...
    'SwarmSize', 50, 'MaxIterations', 200);
[x_ps, fval_ps] = particleswarm(fun, nvars, lb, ub, options_ps);
fprintf('PSO 最优解: %s\n', mat2str(x_ps, 4));
fprintf('PSO 最优值: %.6f\n', fval_ps);

%% 4. TSP 示例（旅行商，用 ga 做整数排列编码）
% 5 个城市坐标
cities = [0 0; 1 2; 2 1; 4 3; 3 0];
n = size(cities, 1);
D = squareform(pdist(cities));   % 距离矩阵
% 用 ga 求解排列（简化示例，实际用 intlinprog 的 MTZ 约束更标准）
fitness_tsp = @(perm) sum(D(sub2ind(size(D), perm, circshift(perm, -1))));
% 这里用启发式改进：模拟退火解 TSP 是经典做法，详见 Python 版 08 脚本
fprintf('城市距离矩阵:\n'); disp(round(D, 2));
fprintf('说明：TSP 的 MATLAB 完整实现可参考 GA 工具箱的 ga(...,' ... 
    'IntCon,...) 整数编码示例，或直接用 Python 版启发式脚本。\n');
