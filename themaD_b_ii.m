%% Levenberg–Marquardt με exact line search
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

% Παράμετροι μεθόδου
mu      = 1.0;     % σταθερή παράμετρος Levenberg–Marquardt
maxIter = 5;
tol_g   = 1e-6;

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
    Bk = H + mu * eye(2);   % τροποποιημένη Hessian (θετικά ορισμένη)

    % Διεύθυνση d_k
    dk = -Bk \ g;

    % Exact line search: ελαχιστοποίηση f(xk + gamma * dk) ως προς gamma >= 0
    phi = @(gamma) f(xk + gamma * dk);   % μονοδιάστατη συνάρτηση
    % Διάστημα αναζήτησης για gamma (π.χ. [0, 5])
    [gamma_k, f_line_min] = fminbnd(phi, 0, 5);

    % Ενημέρωση σημείου
    xk = xk + gamma_k * dk;

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
title('f(x_k) vs k (Levenberg–Marquardt, exact line search, x_0 = (-1,-1))');
