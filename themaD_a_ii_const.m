%% Levenberg–Marquardt για f(x,y) = x^3 * exp(-x^2 - y^4)
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
     x(1).^2 .* x(2).^3 .* (8*x(1).^2 - 12) .* exp(-x(1).^2 - x(2).^4); ...
     x(1).^2 .* x(2).^3 .* (8*x(1).^2 - 12) .* exp(-x(1).^2 - x(2).^4), ...
     x(1).^3 .* x(2).^2 .* (16*x(2).^4 - 12) .* exp(-x(1).^2 - x(2).^4) ];

% Παράμετροι μεθόδου
gamma   = 0.5;   % σταθερό βήμα
mu      = 1.0;   % παράμετρος Levenberg–Marquardt
maxIter = 50;
tol     = 1e-6;

% Αρχικό σημείο
xk = [-1; -1];

% Ιστορία σημείων και τιμών f
X_hist   = xk;        % αποθήκευση x_k
f_values = f(xk);     % f(x_0)

for k = 1:maxIter
    g = gradf(xk);

    % Έλεγχος σύγκλισης
    if norm(g) < tol
        fprintf('Σύγκλιση στο k = %d, x* = [%g  %g]^T, f* = %g\n', ...
                k-1, xk(1), xk(2), f(xk));
        break;
    end

    H  = Hessf(xk);
    Bk = H + mu * eye(2);   % τροποποιημένη Hessian
    dk = -Bk \ g;           % διεύθυνση Levenberg–Marquardt

    % Ενημέρωση με σταθερό γ
    xk = xk + gamma * dk;

    % Αποθήκευση
    X_hist(:,end+1) = xk;
    f_values(end+1) = f(xk);
end

%% Contour plot της f και τροχιάς (x_k, y_k)

% Πλέγμα για την f(x,y)
x1 = linspace(-2, 2, 300);
y1 = linspace(-2, 2, 300);
[X, Y] = meshgrid(x1, y1);
Z = X.^3 .* exp(-X.^2 - Y.^4);

figure;
contour(X, Y, Z, 30);   % ισοϋψείς της f
hold on; grid on;

% Τροχιά
plot(X_hist(1,:), X_hist(2,:), 'ro-', 'LineWidth', 1.5, 'MarkerSize', 5);

% Σημείωση αρχικού και τελικού σημείου
plot(X_hist(1,1),   X_hist(2,1),   'wo', 'MarkerFaceColor','g', 'MarkerSize',8); % αρχή
plot(X_hist(1,end), X_hist(2,end), 'wo', 'MarkerFaceColor','r', 'MarkerSize',8); % τέλος

xlabel('x');
ylabel('y');
title('Πορεία Levenberg–Marquardt με \gamma = 0.5 (αρχή στο (-1,-1))');
axis equal;
hold off;
