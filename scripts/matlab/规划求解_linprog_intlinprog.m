%% ============================================================
% 规划求解_linprog_intlinprog.m
% 用途：线性规划 / 整数规划（MATLAB 版本）
% 依赖：Optimization Toolbox
% ============================================================
clear; clc;

%% 1. 线性规划 linprog
% 例：max z = 3x1 + 5x2
%     s.t.  x1 + 2x2 <= 8
%           3x1 + 2x2 <= 12
%           x1, x2 >= 0
f = [-3; -5];                    % 求 max 取负（linprog 默认求 min）
A = [1 2; 3 2];
b = [8; 12];
lb = [0; 0];                     % 下界
ub = [inf; inf];                 % 上界
[x, fval] = linprog(f, A, b, [], [], lb, ub);
fprintf('LP 最优解: x1=%.4f, x2=%.4f\n', x(1), x(2));
fprintf('LP 最大利润: %.4f\n', -fval);

%% 2. 整数规划 intlinprog
% 例：背包问题 max 8x1+11x2+6x3+4x4
%     s.t. 5x1+7x2+4x3+3x4 <= 14,  xi ∈ {0,1}
f = -[8; 11; 6; 4];
A = [5 7 4 3];
b = 14;
lb = zeros(4, 1);
ub = ones(4, 1);
intcon = 1:4;                    % 全部取整
[x, fval] = intlinprog(f, intcon, A, b, [], [], lb, ub);
fprintf('ILP 最优解: x = %s\n', mat2str(x'));
fprintf('最大价值: %.0f\n', -fval);

%% 3. 指派问题（用 intlinprog 的 0-1 变量）
% 例：3 人 3 任务，成本矩阵 C（行=人，列=任务）
C = [10 15 12; 9 12 11; 13 10 14];
n = size(C, 1);
f = reshape(C, [], 1);           % 变量 x_ij 按列展开
% 每行和为 1（每人一个任务）、每列和为 1（每个任务一人做）
Aeq = [kron(eye(n), ones(1, n)); kron(ones(1, n), eye(n))];
beq = ones(2 * n, 1);
lb = zeros(n * n, 1); ub = ones(n * n, 1); intcon = 1:n*n;
[x, fval] = intlinprog(f, intcon, [], [], Aeq, beq, lb, ub);
X = reshape(x, n, n);
fprintf('指派方案（行=人,列=任务）:\n'); disp(X);
fprintf('最小总成本: %.0f\n', fval);
