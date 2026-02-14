% Μέγιστη Καθόδου με γ = 0.1 και αρχικό σημείο (1,1)
clear; clc;

% Ορισμός της συνάρτησης και gradient
syms x y
f = x^3 * exp(-x^2 - y^4);
grad_f = gradient(f, [x, y]);

% Παράμετροι
gamma = 0.2;
xk = [1; 1];
eps = 1e-6;
max_iter = 100;
path = xk.';

% Μέθοδος μέγιστης καθόδου
for k = 1:max_iter
    gk = double(subs(grad_f, {x, y}, {xk(1), xk(2)}));
    
    if norm(gk) < eps
        break;
    end
    
    dk = -gk;
    xk = xk + gamma * dk;
    path(end+1, :) = xk.';
end

% Διάγραμμα ισοϋψών και πορεία
f_func = matlabFunction(f, 'Vars', {x, y});
[X, Y] = meshgrid(-2:0.05:2, -2:0.05:2);
Z = f_func(X, Y);

figure;
contour(X, Y, Z, 30); hold on;
plot(path(:,1), path(:,2), 'r-o', 'LineWidth', 2, 'MarkerSize', 4);
xlabel('x'); ylabel('y');
title('Πορεία Μέγιστης Καθόδου με γ = 0.2 (αρχή στο (1,1))');
grid on;
