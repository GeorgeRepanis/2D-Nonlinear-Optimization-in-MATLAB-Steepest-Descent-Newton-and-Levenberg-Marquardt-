clear; clc; close all;

%% Συνάρτηση, gradient, Hessian
f = @(x) x(1).^3 .* exp(-x(1).^2 - x(2).^4);

gradf = @(x) [ ...
    (-2*x(1).^4 + 3*x(1).^2) .* exp(-x(1).^2 - x(2).^4);   % df/dx
    -4*x(1).^3 .* x(2).^3 .* exp(-x(1).^2 - x(2).^4)       % df/dy
];

Hessf = @(x) [ ...
    (4*x(1).^5 - 14*x(1).^3 + 6*x(1)) .* exp(-x(1).^2 - x(2).^4), ...
    (8*x(1).^4.*x(2).^3 - 12*x(1).^2.*x(2).^3) .* exp(-x(1).^2 - x(2).^4); ...
    (8*x(1).^4.*x(2).^3 - 12*x(1).^2.*x(2).^3) .* exp(-x(1).^2 - x(2).^4), ...
    (16*x(1).^3.*x(2).^6 - 12*x(1).^3.*x(2).^2) .* exp(-x(1).^2 - x(2).^4) ...
];

%% Παράμετροι Armijo
alpha = 1e-4;    % 0 < alpha < 1
beta  = 0.5;     % 0 < beta < 1
s     = 1;       % αρχικό βήμα

%% Newton + Armijo από x0 = (1,1)
xk = [1; 1];
maxIter = 50;
tol = 1e-6;

f_vals = zeros(maxIter+1,1);
f_vals(1) = f(xk);

k_last = 0;

for k = 1:maxIter
    gk = gradf(xk);
    if norm(gk) < tol
        k_last = k-1;
        break;
    end

    Hk = Hessf(xk);

    % ---- Διεύθυνση Newton με safeguard ----
    if rcond(Hk) < 1e-12
        dk = -gk;            % fallback σε steepest descent
    else
        dk = -Hk \ gk;       % κανονική διεύθυνση Newton
        if gk.'*dk >= 0      % αν δεν είναι κατεύθυνση καθόδου
            dk = -gk;
        end
    end

    % ---- Armijo backtracking ----
    gamma = s;
    fk = f(xk);
    while f(xk + gamma*dk) > fk + alpha*gamma*(gk.'*dk)
        gamma = beta * gamma;
        if gamma < 1e-10
            break;
        end
    end

    % ενημέρωση
    xk = xk + gamma*dk;

    f_vals(k+1) = f(xk);
    k_last = k;
end

% αν δεν έγινε break, τότε k_last = maxIter ήδη

k_vec = 0:k_last;

%% Διάγραμμα σύγκλισης f(x_k)
figure;
plot(k_vec, f_vals(1:k_last+1), '-o', 'LineWidth', 2, 'MarkerSize', 6);
grid on;
xlabel('Επαναλήψεις (k)', 'FontSize', 12);
ylabel('f(x_k)', 'FontSize', 12);
title('Σύγκλιση της f με Newton , Armijo από το (1,1)', 'FontSize', 14);
set(gca, 'FontSize', 11);
