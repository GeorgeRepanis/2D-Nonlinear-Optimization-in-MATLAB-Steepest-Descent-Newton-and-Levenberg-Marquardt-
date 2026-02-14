%% Levenberg–Marquardt με κανόνα Armijo
% f(x,y) = x^3 * exp(-x^2 - y^4), x0 = (-1,-1)

clear; clc;

% Αντικειμενική συνάρτηση
f = @(x) x(1).^3 .* exp(-x(1).^2 - x(2).^4);

% Κλίση
gradf = @(x) [ ...
    x(1).^2 .* (3 - 2*x(1).^2) .* exp(-x(1).^2 - x(2).^4); ...
   -4*x(1).^3 .* x(2).^3 .* exp(-x(1).^2 - x(2).^4)       ];

% Hessian
Hessf = @(x) [ ...
    (4*x(1).^5 - 14*x(1).^3 + 6*x(1)) .* exp(-x(1).^2 - x(2).^4), ...
    (8*x(1).^4 .* x(2).^3 - 12*x(1).^2 .* x(2).^3) .* exp(-x(1).^2 - x(2).^4); ...
    (8*x(1).^4 .* x(2).^3 - 12*x(1).^2 .* x(2).^3) .* exp(-x(1).^2 - x(2).^4), ...
    (16*x(1).^3 .* x(2).^6 - 12*x(1).^3 .* x(2).^2) .* exp(-x(1).^2 - x(2).^4) ];

% Παράμετροι Levenberg–Marquardt
mu      = 1.0;      % σταθερή παράμετρος μ
maxIter = 10;
tol_g   = 1e-6;

% Παράμετροι κανόνα Armijo
alpha = 1e-1;       % 0 < alpha < 1
beta  = 0.5;        % 0 < beta < 1
s     = 1.0;        % αρχικό βήμα

% Αρχικό σημείο
xk = [-1; -1];

% Αποθήκευση f(x_k)
f_values = f(xk);   % f(x_0)

for k = 1:maxIter
    g = gradf(xk);

    % Έλεγχος σύγκλισης με βάση την κλίση
    if norm(g) < tol_g
        fprintf('Σύγκλιση στο k = %d, x* = [%g  %g]^T, f* = %g\n', ...
                k-1, xk(1), xk(2), f(xk));
        break;
    end

    % Πίνακας Levenberg–Marquardt
    H  = Hessf(xk);
    Bk = H + mu * eye(2);       % θετικά ορισμένος (με κατάλληλο μ)

    % Διεύθυνση d_k
    dk = -Bk \ g;

    % Αν η διεύθυνση είναι "πολύ μικρή", σταμάτα
    if norm(dk) < 1e-12
        break;
    end

    % --- Κανόνας Armijo για το γ_k ---
    gamma = s;
    while true
        x_trial = xk + gamma * dk;
        if f(x_trial) <= f(xk) + alpha * gamma * (g.' * dk)
            break;         % βρέθηκε κατάλληλο γ
        end
        gamma = beta * gamma;
        if gamma < 1e-8    % πολύ μικρό βήμα -> σταματάμε
            break;
        end
    end
    % -------------------------------

    % Ενημέρωση σημείου
    xk = xk + gamma * dk;

    % Αποθήκευση f(x_k)
    f_values(end+1,1) = f(xk);
end

% Δείκτης επαναλήψεων
k_vec = 0:length(f_values)-1;

% Γράφημα f(x_k) ως προς k
figure;
plot(k_vec, f_values, 'o-', 'LineWidth', 1.5);
grid on;
xlabel('k');
ylabel('f(x_k)');
title('f(x_k) συναρτήσει των επαναλήψεων (Levenberg–Marquardt, Armijo, x_0 = (-1,-1))');
