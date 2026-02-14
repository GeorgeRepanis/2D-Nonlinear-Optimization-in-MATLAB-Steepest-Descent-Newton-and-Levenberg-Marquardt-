%% Steepest Descent για f(x,y) = x^3 e^{-x^2-y^4}
clear; clc; close all;

% Συνάρτηση και gradient
f  = @(x,y) x.^3 .* exp(-x.^2 - y.^4);
gx = @(x,y) (3*x.^2 - 2*x.^4) .* exp(-x.^2 - y.^4);
gy = @(x,y) -4*x.^3 .* y.^3 .* exp(-x.^2 - y.^4);

% Αρχικό σημείο και βήμα
xk = [1; 1];          % x0 = (1,1)
gamma = 0.2;          % σταθερό βήμα
maxIter = 15;        % πλήθος επαναλήψεων

% Πίνακας τιμών f(x_k)
fk = zeros(maxIter+1,1);
fk(1) = f(xk(1), xk(2));    % τιμή στο x0

% Επαναλήψεις Steepest Descent
for k = 1:maxIter
    gk = [gx(xk(1), xk(2));
          gy(xk(1), xk(2))];      % gradient ∇f(x_k)
    dk = -gk;                     % διεύθυνση μέγιστης καθόδου
    xk = xk + gamma*dk;           % νέο σημείο x_{k+1}
    fk(k+1) = f(xk(1), xk(2));    % αποθήκευση f(x_{k+1})
end

% Πραγματικό τοπικό ελάχιστο (στην κοιλάδα γύρω από x = -sqrt(3/2), y = 0)
x_star = -sqrt(3/2);
y_star = 0;
f_true_min = f(x_star, y_star);

k_vec = 0:maxIter;

%% Γράφημα f(x_k) vs k (όπως στην εικόνα)
figure;
plot(k_vec, fk, 'o-', 'LineWidth', 1.5, 'MarkerSize', 4); hold on;

% Οριζόντια γραμμή του πραγματικού τοπικού ελαχίστου
yline(f_true_min, '--', 'f_{true min} \approx -0.41', ...
    'LineWidth', 1.2, 'LabelHorizontalAlignment', 'right');

% Όρια αξόνων ώστε να ταιριάζουν με το διάγραμμα
xlim([0 maxIter]);
ylim([-0.5 0.2]);

% Τίτλοι/Άξονες
title('f(x_k) για Steepest Descent με x_0 = (1,1) και \gamma = 0.2');
xlabel('Επαναλήψεις k');
ylabel('f(x_k)');

% Legend
legend({'f(x_k)', 'Πραγματικό τοπικό ελάχιστο'}, ...
       'Location', 'southeast');

% Στυλ σκοτεινού φόντου όπως στο παράδειγμά σου
set(gcf, 'Color', 'k');           % φόντο figure
ax = gca;
ax.Color   = 'k';                 % φόντο αξόνων
ax.XColor  = 'w';                 % χρώμα αξόνων
ax.YColor  = 'w';
ax.GridColor = [0.5 0.5 0.5];
grid on;
