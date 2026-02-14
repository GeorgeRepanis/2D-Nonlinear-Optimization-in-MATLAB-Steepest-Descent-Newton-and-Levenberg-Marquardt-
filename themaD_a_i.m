%% Levenberg–Marquardt για f(x,y) = x^3 * exp(-x^2 - y^4)
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
     x(1).^2 .* x(2).^3 .* (8*x(1).^2 - 12) .* exp(-x(1).^2 - x(2).^4); ...
     x(1).^2 .* x(2).^3 .* (8*x(1).^2 - 12) .* exp(-x(1).^2 - x(2).^4), ...
     x(1).^3 .* x(2).^2 .* (16*x(2).^4 - 12) .* exp(-x(1).^2 - x(2).^4) ];

% Παράμετροι μεθόδου
gamma   = 0.5;   % σταθερό βήμα
mu      = 1.0;   % παράμετρος Levenberg–Marquardt
maxIter = 50;
tol     = 1e-6;

% Αρχικό σημείο
x = [0; 0];

% Αποθήκευση τιμών f(x_k)
f_values = f(x);      % f(x_0)
xk       = x;

for k = 1:maxIter
    g = gradf(xk);

    % Έλεγχος σύγκλισης (στο (0,0) ισχύει από την αρχή)
    if norm(g) < tol
        fprintf('Σύγκλιση στο k = %d, x* = [%g  %g]^T\n', ...
                k-1, xk(1), xk(2));
        break;
    end

    H  = Hessf(xk);
    Bk = H + mu * eye(2);      % Τροποποιημένη Hessian
    dk = -Bk \ g;              % Διεύθυνση Levenberg–Marquardt

    % Ενημέρωση
    xk = xk + gamma * dk;

    % Αποθήκευση f(x_k)
    f_values(end+1,1) = f(xk);
end

% Δείκτης επαναλήψεων
k_vec = 0:length(f_values)-1;

% Γράφημα f(x_k) ως προς k
figure;
plot(k_vec, f_values, 'o-','LineWidth',1.5);
grid on;
xlabel('k');
ylabel('f(x_k)');
title('f(x_k) συναρτήσει των επαναλήψεων (Levenberg–Marquardt, \gamma = 0.5)');
