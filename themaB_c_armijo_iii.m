clc; clear;

% Αρχικές τιμές
x = [1; 1];             % αρχικό σημείο (x0, y0)
s = 1;                  % αρχικό βήμα
beta = 0.5;             % παράγοντας μείωσης
sigma = 1e-4;           % παράμετρος Armijo
max_iter = 100;         % μέγιστος αριθμός επαναλήψεων
tol = 1e-6;             % ανοχή για σύγκλιση

% Συνάρτηση
f = @(x) x(1)^3 * exp(-x(1)^2 - x(2)^4);

% Gradient της f
gradf = @(x) [x(1)^2*(3 - 2*x(1)^2)*exp(-x(1)^2 - x(2)^4);
              -4*x(1)^3*x(2)^3*exp(-x(1)^2 - x(2)^4)];

% Αποθήκευση διαδρομής
trajectory = x';

for k = 1:max_iter
    gk = gradf(x);         % gradient
    dk = -gk;              % κατεύθυνση καθόδου

    % Υπολογισμός βήματος γ μέσω backtracking Armijo
    gamma = s;
    while f(x + gamma * dk) > f(x) + sigma * gamma * gk' * dk
        gamma = beta * gamma;
    end

    % Ενημέρωση σημείου
    x_new = x + gamma * dk;
    trajectory(end+1, :) = x_new';

    % Έλεγχος σύγκλισης
    if norm(x_new - x) < tol
        break;
    end

    x = x_new;
end

% Εμφάνιση αποτελεσμάτων
disp(['Τελικό σημείο: x = ', num2str(x(1)), ', y = ', num2str(x(2))]);
disp(['Τελική τιμή f(x, y): ', num2str(f(x))]);

% Προαιρετικό plot της διαδρομής
[X, Y] = meshgrid(-2:0.05:2, -2:0.05:2);
Z = X.^3 .* exp(-X.^2 - Y.^4);
figure;
contour(X, Y, Z, 50); hold on;
plot(trajectory(:,1), trajectory(:,2), 'r.-', 'LineWidth', 1.5);
title('Πορεία Μέγιστης Καθόδου με Κανόνα Armijo (αρχή στο (1,1))');
xlabel('x'); ylabel('y');
