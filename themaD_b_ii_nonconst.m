%% Πορεία Levenberg–Marquardt με Exact Line Search (x0 = (-1,-1))
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

% Παράμετροι μεθόδου
mu      = 1.0;      % σταθερή παράμετρος Levenberg–Marquardt
maxIter = 50;
tol_g   = 1e-6;

% Αρχικό σημείο
xk = [-1; -1];

% Ιστορία σημείων
X_hist = xk;

for k = 1:maxIter
    g = gradf(xk);

    % Έλεγχος σύγκλισης
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

    % Exact line search: γ_k = argmin_{γ>=0} f(xk + γ d_k)
    phi = @(gamma) f(xk + gamma * dk);
    [gamma_k, ~] = fminbnd(phi, 0, 5);    % διάστημα αναζήτησης για γ

    % Ενημέρωση
    xk = xk + gamma_k * dk;

    % Αποθήκευση τροχιάς
    X_hist(:, end+1) = xk;
end

%% Contour plot της f και τροχιάς (x_k, y_k)

% Πλέγμα για την f(x,y)
x1 = linspace(-2, 2, 400);
y1 = linspace(-2, 2, 400);
[X, Y] = meshgrid(x1, y1);
Z = X.^3 .* exp(-X.^2 - Y.^4);

figure;
contour(X, Y, Z, 30);  % ισοϋψείς της f
hold on; grid on;

% Τροχιά
plot(X_hist(1,:), X_hist(2,:), 'ro-', 'LineWidth', 1.5, 'MarkerSize', 5);

% Αρχικό και τελικό σημείο
plot(X_hist(1,1),   X_hist(2,1),   'wo', 'MarkerFaceColor','g', 'MarkerSize',8); % αρχή
plot(X_hist(1,end), X_hist(2,end), 'wo', 'MarkerFaceColor','r', 'MarkerSize',8); % τέλος

xlabel('x');
ylabel('y');
title('Πορεία Levenberg–Marquardt με Exact Line Search (αρχή στο (-1,-1))');
axis equal;
hold off;
