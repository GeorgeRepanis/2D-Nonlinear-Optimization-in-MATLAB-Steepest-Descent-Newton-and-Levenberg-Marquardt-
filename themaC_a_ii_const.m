% Μέθοδος Newton με σταθερό γ = 0.5
clear; clc;

% Ορισμός της συνάρτησης και παραγώγων
syms x y
f = x^3 * exp(-x^2 - y^4);
grad_f = gradient(f, [x, y]);
H_f = hessian(f, [x, y]);

% Παράμετροι
gamma = 0.5;
xk = [-1; -1];  % αρχικό σημείο
eps = 1e-6;
max_iter = 100;
path = xk.';

% Επανάληψη Newton
for k = 1:max_iter
    gk = double(subs(grad_f, {x, y}, {xk(1), xk(2)}));
    Hk = double(subs(H_f, {x, y}, {xk(1), xk(2)}));
    
    if norm(gk) < eps
        break;
    end
    
    pk = -Hk \ gk;
    xk = xk + gamma * pk;
    path(end+1, :) = xk.';
end

% Διάγραμμα ισοϋψών και πορείας
f_func = matlabFunction(f, 'Vars', {x, y});
[X, Y] = meshgrid(-2:0.05:2, -2:0.05:2);
Z = f_func(X, Y);

figure;
contour(X, Y, Z, 30); hold on;
plot(path(:,1), path(:,2), 'r-o', 'LineWidth', 2, 'MarkerSize', 4);
xlabel('x'); ylabel('y');
title('Πορεία Μεθόδου Newton με γ = 0.5 (αρχή στο (-1,-1))');
grid on;
