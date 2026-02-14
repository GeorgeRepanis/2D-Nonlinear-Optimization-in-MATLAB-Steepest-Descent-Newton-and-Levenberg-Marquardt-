%% Levenberg–Marquardt με κανόνα Armijo
% f(x,y) = x^3 * exp(-x^2 - y^4), x0 = (1,1)

clear; clc;

% ===== Συνάρτηση, κλίση, Hessian =====
f = @(x) x(1).^3 .* exp(-x(1).^2 - x(2).^4);

gradf = @(x) [ ...
    x(1).^2 .* (3 - 2*x(1).^2) .* exp(-x(1).^2 - x(2).^4); ...
   -4*x(1).^3 .* x(2).^3 .* exp(-x(1).^2 - x(2).^4)       ];

Hessf = @(x) [ ...
    (4*x(1).^5 - 14*x(1).^3 + 6*x(1)) .* exp(-x(1).^2 - x(2).^4), ...
    (8*x(1).^4 .* x(2).^3 - 12*x(1).^2 .* x(2).^3) .* exp(-x(1).^2 - x(2).^4); ...
    (8*x(1).^4 .* x(2).^3 - 12*x(1).^2 .* x(2).^3) .* exp(-x(1).^2 - x(2).^4), ...
    (16*x(1).^3 .* x(2).^6 - 12*x(1).^3 .* x(2).^2) .* exp(-x(1).^2 - x(2).^4) ];

% ===== Παράμετροι Levenberg–Marquardt & Armijo =====
mu      = 1.0;      % σταθερή παράμετρος μ
maxIter = 50;
tol_g   = 1e-6;

alpha = 1e-1;       % Armijo: 0 < alpha < 1
beta  = 0.5;        % Armijo: 0 < beta < 1
s     = 1.0;        % αρχικό βήμα για backtracking

% ===== Αρχικό σημείο =====
xk = [1; 1];

% Ιστορία τιμών και σημείων
f_values = f(xk);      % f(x_0)
X_hist   = xk;         % τροχιά (x_k, y_k)

% ===== Βρόχος επαναλήψεων =====
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
    Bk = H + mu * eye(2);

    % Διεύθυνση d_k
    dk = -Bk \ g;

    if norm(dk) < 1e-12
        break;   % πάρα πολύ μικρή διεύθυνση
    end

    % ===== Κανόνας Armijo για γ_k =====
    gamma = s;
    while true
        x_trial = xk + gamma * dk;
        if f(x_trial) <= f(xk) + alpha * gamma * (g.' * dk)
            break;   % το βήμα είναι αποδεκτό
        end
        gamma = beta * gamma;
        if gamma < 1e-8
            break;   % πολύ μικρό βήμα -> σταματάμε
        end
    end
    % ==================================

    % Ενημέρωση σημείου
    xk = xk + gamma * dk;

    % Αποθήκευση
    f_values(end+1,1) = f(xk);
    X_hist(:, end+1)  = xk;
end

% ===== Διάγραμμα f(x_k) vs k =====
k_vec = 0:length(f_values)-1;

figure;
plot(k_vec, f_values, 'o-', 'LineWidth', 1.5);
grid on;
xlabel('k');
ylabel('f(x_k)');
title('f(x_k) συναρτήσει των επαναλήψεων (Levenberg–Marquardt, Armijo, x_0 = (1,1))');

% ===== Contour + τροχιά στο (x,y) =====
x1 = linspace(-2, 2, 400);
y1 = linspace(-2, 2, 400);
[X, Y] = meshgrid(x1, y1);
Z = X.^3 .* exp(-X.^2 - Y.^4);

figure;
contour(X, Y, Z, 30);
hold on; grid on;

plot(X_hist(1,:), X_hist(2,:), 'ro-', 'LineWidth', 1.5, 'MarkerSize', 5);
plot(X_hist(1,1),   X_hist(2,1),   'wo', 'MarkerFaceColor','g', 'MarkerSize',8); % αρχή
plot(X_hist(1,end), X_hist(2,end), 'wo', 'MarkerFaceColor','r', 'MarkerSize',8); % τέλος

xlabel('x'); ylabel('y');
title('Πορεία Levenberg–Marquardt με Armijo (αρχή στο (1,1))');
axis equal;
hold off;
