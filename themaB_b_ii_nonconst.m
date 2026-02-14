% Gradient Descent με Exact Line Search (από σημειώσεις)
clear; clc;

% Ορισμός συμβολικά
syms x y gamma
f = x^3 * exp(-x^2 - y^4);
grad_f = gradient(f, [x, y]);

% Παράμετροι
xk = [-1; -1];
eps = 1e-6;
max_iter = 100;
path = xk.';

% Συνάρτηση f ως MATLAB function
f_func = matlabFunction(f, 'Vars', {x, y});

for k = 1:max_iter
    gk = double(subs(grad_f, {x, y}, {xk(1), xk(2)}));
    
    if norm(gk) < eps
        break;
    end

    dk = -gk;
    
    % Ορισμός της φ_k(γ) για fixed xk και dk
    phi = @(gamma) f_func(xk(1) + gamma * dk(1), xk(2) + gamma * dk(2));
    
    % Ελαχιστοποίηση με fminbnd στο [0, 0.5]
    gamma_k = fminbnd(phi, 0, 0.5);
    
    % Ενημέρωση
    xk = xk + gamma_k * dk;
    path(end+1, :) = xk.';
end

% Διάγραμμα ισοϋψών + πορεία
[X, Y] = meshgrid(-2:0.05:2, -2:0.05:2);
Z = f_func(X, Y);

figure;
contour(X, Y, Z, 30); hold on;
plot(path(:,1), path(:,2), 'r-o', 'LineWidth', 2, 'MarkerSize', 4);
xlabel('x'); ylabel('y');
title('Πορεία Μέγιστης Καθόδου με Exact Line Search (αρχή στο (-1,-1))');
grid on;
