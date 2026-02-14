%% f(x_k) vs k για Levenberg–Marquardt με κανόνα Armijo
% f(x,y) = x^3 * exp(-x^2 - y^4), x0 = (1,1)

clear; clc;

% ---- Συνάρτηση, κλίση, Hessian ----
f = @(x) x(1).^3 .* exp(-x(1).^2 - x(2).^4);

gradf = @(x) [ ...
    x(1).^2 .* (3 - 2*x(1).^2) .* exp(-x(1).^2 - x(2).^4); ...
   -4*x(1).^3 .* x(2).^3 .* exp(-x(1).^2 - x(2).^4)       ];

Hessf = @(x) [ ...
    (4*x(1).^5 - 14*x(1).^3 + 6*x(1)) .* exp(-x(1).^2 - x(2).^4), ...
    (8*x(1).^4 .* x(2).^3 - 12*x(1).^2 .* x(2).^3) .* exp(-x(1).^2 - x(2).^4); ...
    (8*x(1).^4 .* x(2).^3 - 12*x(1).^2 .* x(2).^3) .* exp(-x(1).^2 - x(2).^4), ...
    (16*x(1).^3 .* x(2).^6 - 12*x(1).^3 .* x(2).^2) .* exp(-x(1).^2 - x(2).^4) ];

% ---- Παράμετροι Levenberg–Marquardt & Armijo ----
mu      = 1.0;      % σταθερή μ
maxIter = 5;
tol_g   = 1e-6;

alpha = 1e-1;       % Armijo
beta  = 0.5;
s     = 1.0;

% ---- Αρχικό σημείο ----
xk = [1; 1];

% Ιστορία f(x_k)
f_values = f(xk);     % f(x_0)

% ---- Επαναλήψεις ----
for k = 1:maxIter
    g = gradf(xk);
    if norm(g) < tol_g
        break;
    end

    H  = Hessf(xk);
    Bk = H + mu * eye(2);
    dk = -Bk \ g;

    if norm(dk) < 1e-12
        break;
    end

    % Armijo για γ_k
    gamma = s;
    while true
        x_trial = xk + gamma * dk;
        if f(x_trial) <= f(xk) + alpha * gamma * (g.' * dk)
            break;
        end
        gamma = beta * gamma;
        if gamma < 1e-8
            break;
        end
    end

    % Ενημέρωση
    xk = xk + gamma * dk;
    f_values(end+1,1) = f(xk);
end

% ---- Διάγραμμα f(x_k) vs k ----
k_vec = 0:length(f_values)-1;

figure;
plot(k_vec, f_values, 'o-', 'LineWidth', 1.5);
grid on;
xlabel('k');
ylabel('f(x_k)');
title('f(x_k) συναρτήσει των επαναλήψεων (Levenberg–Marquardt, Armijo, x_0 = (1,1))');
