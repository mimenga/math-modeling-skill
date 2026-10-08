%% ============================================================
% 拟合与插值_polyfit_interp.m
% 用途：数据拟合 / 插值（只有离散数据时补中间值、算积分）
% ============================================================
clear; clc;
rng(42);                          % 固定种子，可复现

%% 1. 多项式拟合 polyfit（最小二乘）
x = linspace(0, 10, 50)';
y = 0.5*x.^2 + 2*x + 3 + 5*randn(50, 1);   % 带噪声的二次数据
p = polyfit(x, y, 2);                      % 2 阶多项式系数
fprintf('拟合多项式系数: a=%.3f, b=%.3f, c=%.3f\n', p(1), p(2), p(3));
yfit = polyval(p, x);
R2 = 1 - sum((y - yfit).^2) / sum((y - mean(y)).^2);
fprintf('拟合 R^2 = %.4f\n', R2);
figure;
plot(x, y, 'o', x, yfit, 'r-', 'LineWidth', 1.5);
legend('数据', '拟合曲线'); title('多项式拟合'); grid on;

%% 2. 插值 interp1（精确过点）
% 例：对稀疏数据补中间值
xs = [0 1 2 3 4 5]';
ys = [0 0.8 2.2 3.1 3.9 5.5]';
xq = 0:0.1:5;
yq_linear = interp1(xs, ys, xq, 'linear');     % 线性插值
yq_spline = interp1(xs, ys, xq, 'spline');     % 样条插值（平滑）
figure;
plot(xs, ys, 'ko', xq, yq_linear, 'b--', xq, yq_spline, 'r-', 'LineWidth', 1.5);
legend('原始点', '线性插值', '样条插值'); title('插值对比'); grid on;

%% 3. 数值积分（梯形/Simpson）
% 例：用离散数据算积分（如由浓度曲线算总量）
% 梯形法
integral_trap = trapz(xq, yq_spline);
% Simpson 法（需偶数个区间）
n = length(xq);
integral_simpson = (xq(end)-xq(1))/(3*(n-1)) * ...
    (yq_spline(1) + yq_spline(end) + 4*sum(yq_spline(2:2:end-1)) ...
     + 2*sum(yq_spline(3:2:end-2)));
fprintf('梯形积分 = %.4f, Simpson 积分 = %.4f\n', ...
    integral_trap, integral_simpson);

%% 4. 数值微分（由离散点算导数/变化率）
% 例：由浓度数据算变化率
dy = diff(yq_spline) ./ diff(xq);
figure;
plot(xq(1:end-1), dy, 'b-', 'LineWidth', 1.5);
title('数值微分（变化率）'); grid on;
