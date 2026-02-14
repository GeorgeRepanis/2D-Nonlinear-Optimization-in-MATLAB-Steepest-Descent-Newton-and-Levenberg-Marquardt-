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
f_vals = f_func(xk(1), xk(2));

for k = 1:max_iter
    gk = double(subs(grad_f, {x, y}, {xk(1), xk(2)}));
    Hk = double(subs(H_f, {x, y}, {xk(1), xk(2)}));

    if norm(gk) < eps
        break;
    end

    if det(Hk) == 0
        warning('Μη αντιστρέψιμο Hessian');
        break;
    end

    pk = -Hk \ gk;

    % Exact line search
    phi = @(gamma) f_func(xk(1) + gamma * pk(1), xk(2) + gamma * pk(2));
    gamma_k = fminbnd(phi, 0, 1);  % επέκταση διαστήματος

    xk = xk + gamma_k * pk;
    f_new = f_func(xk(1), xk(2));
    f_vals(end+1) = f_new;

    % Κριτήριο σύγκλισης τιμών f
    if abs(f_vals(end) - f_vals(end-1)) < eps
        break;
    end
end

% Διάγραμμα
figure;
plot(0:length(f_vals)-1, f_vals, '-o', 'LineWidth', 2);
xlabel('Επαναλήψεις (k)');
ylabel('f(x_k)');
title('Σύγκλιση της f με Newton και Exact Line Search');
grid on;
