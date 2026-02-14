clear; clc;

% Συμβολικός ορισμός
syms x y
f_sym = x^3 * exp(-x^2 - y^4);
grad_f = gradient(f_sym, [x, y]);
H_f = hessian(f_sym, [x, y]);

% Αρχικό σημείο
xk = [-1; -1];
eps = 1e-8;
max_iter = 100;
f_func = matlabFunction(f_sym, 'Vars', {x, y});
path = xk.';

for k = 1:max_iter
    gk = double(subs(grad_f, {x, y}, {xk(1), xk(2)}));
    Hk = double(subs(H_f, {x, y}, {xk(1), xk(2)}));

    if norm(gk) < eps || det(Hk) == 0
        break;
    end

    pk = -Hk \ gk;

    % Exact line search με ελαχιστοποίηση f(x + γ p)
    phi = @(gamma) f_func(xk(1) + gamma * pk(1), xk(2) + gamma * pk(2));
    gamma_k = fminbnd(phi, 0, 1);

    % Ενημέρωση
    xk = xk + gamma_k * pk;
    path(end+1, :) = xk.';
    
    % Κριτήριο σύγκλισης
    if norm(pk) < eps
        break;
    end
end

% Γραφικό με ισοκαμπύλες και πορεία
[X, Y] = meshgrid(-2:0.05:2, -2:0.05:2);
F = @(x, y) x.^3 .* exp(-x.^2 - y.^4);
Z = F(X, Y);

figure;
contour(X, Y, Z, 30);
hold on;
plot(path(:,1), path(:,2), 'ro-', 'LineWidth', 2, 'MarkerSize', 5);
xlabel('x'); ylabel('y');
title('Πορεία Newton με Exact Line Search (αρχή στο (-1,-1))');
grid on;
