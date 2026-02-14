clc; clear;

% Συνάρτηση και παράγωγοι
f = @(x) x(1)^3 * exp(-x(1)^2 - x(2)^4);
gradf = @(x) [...
    (3*x(1)^2 - 2*x(1)^4) * exp(-x(1)^2 - x(2)^4); 
    -4*x(2)^3 * x(1)^3 * exp(-x(1)^2 - x(2)^4)];
Hf = @(x) [...
    (6*x(1) - 14*x(1)^3 + 4*x(1)^5) * exp(-x(1)^2 - x(2)^4), ...
    -12*x(2)^3 * x(1)^2 * (1 - x(1)^2) * exp(-x(1)^2 - x(2)^4);
    -12*x(2)^3 * x(1)^2 * (1 - x(1)^2) * exp(-x(1)^2 - x(2)^4), ...
    -4*x(1)^3 * exp(-x(1)^2 - x(2)^4) * (3*x(2)^2 - 4*x(2)^6)];

% Αρχικό σημείο
x0 = [1; 1];
xk = x0;
X = xk';              % για plotting διαδρομής
fk = f(xk);
F = fk;               % για f vs k

maxit = 100;
tol = 1e-6;

for k = 1:maxit
    gk = gradf(xk);
    Hk = Hf(xk);
    
    % Κατεύθυνση Newton (προσοχή: gk πρέπει να είναι διάνυσμα στήλης)
    dk = -Hk \ gk(:);
    
    % Exact Line Search: ελαχιστοποίηση f(xk + γ dk)
    phi = @(g) f(xk + g * dk);
    g_opt = fminbnd(phi, 0, 1);  % Περιορισμός του search

    % Ενημέρωση
    xk = xk + g_opt * dk;
    
    X = [X; xk'];
    fk = f(xk);
    F = [F; fk];
    
    % Έλεγχος σύγκλισης
    if norm(gk) < tol
        break;
    end
end

% --- Γράφημα πορείας στο επίπεδο ---
figure;
[X1, X2] = meshgrid(linspace(-2,2,100), linspace(-2,2,100));
Z = arrayfun(@(x,y) f([x;y]), X1, X2);
contour(X1, X2, Z, 30); hold on;
plot(X(:,1), X(:,2), 'r.-', 'LineWidth', 1.5, 'MarkerSize', 10)
title('Πορεία Newton με Exact Line Search (αρχή στο (1,1))', 'FontWeight','bold')
xlabel('x'); ylabel('y');

% --- f(x_k) vs k ---
figure;
plot(0:length(F)-1, F, 'o-', 'LineWidth', 1.5)
title('Σύγκλιση της f με Newton και Exact Line Search από το (1,1) ', 'FontWeight','bold')
xlabel('Επαναλήψεις (k)');
ylabel('f(x_k)');
grid on;
