%% Πορεία Levenberg–Marquardt με κανόνα Armijo (x0 = (-1,-1))
clear; clc;

% Συνάρτηση
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
maxIter = 50;
tol_g   = 1e-6;

% Παράμετροι Armijo
alpha = 1e-1;
beta  = 0.5;
s     = 1.0;

% Αρχικό σημείο
xk = [-1; -1];

% Ιστορία σημείων
X_hist = xk;

for k = 1:maxIter
    g = gradf(xk);
    if norm(g) < tol_g
        fprintf('Σύγκλιση στο k = %d, x* = [%g  %g]^T, f* = %g\n', ...
                k-1, xk(1), xk(2), f(xk));
        break;
    end

    H  = Hessf(xk);
    Bk = H + mu * eye(2);
    dk = -Bk \ g;

    if norm(dk) < 1e-12
        break;
    end

    % --- Κανόνας Armijo ---
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
    % ----------------------

    xk = xk + gamma * dk;
    X_hist(:, end+1) = xk;
end

%% Contour plot της f και τροχιάς (x_k, y_k)

x1 = linspace(-2, 2, 400);
y1 = linspace(-2, 2, 400);
[X, Y] = meshgrid(x1, y1);
Z = X.^3 .* exp(-X.^2 - Y.^4);

figure;
contour(X, Y, Z, 30);            % ισοϋψείς της f
hold on; grid on;

% Τροχιά
plot(X_hist(1,:), X_hist(2,:), 'ro-', 'LineWidth', 1.5, 'MarkerSize', 5);

% Αρχικό (πράσινο) & τελικό (κόκκινο) σημείο
plot(X_hist(1,1),   X_hist(2,1),   'wo', 'MarkerFaceColor','g', 'MarkerSize',8);
plot(X_hist(1,end), X_hist(2,end), 'wo', 'MarkerFaceColor','r', 'MarkerSize',8);

xlabel('x'); ylabel('y');
title('Πορεία Levenberg–Marquardt με Armijo (αρχή στο (-1,-1))');
axis equal;
hold off;
