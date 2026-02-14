clear; clc; close all;

% Συνάρτηση
f = @(x) x(1)^3 * exp(-x(1)^2 - x(2)^4);

% Gradient
gradf = @(x) [ ...
    (3*x(1)^2 - 2*x(1)^4) * exp(-x(1)^2 - x(2)^4); ...
    -4 * x(1)^3 * x(2)^3 * exp(-x(1)^2 - x(2)^4)];

% Hessian
Hessf = @(x) [
    (6*x(1) - 14*x(1)^3 + 4*x(1)^5) * exp(-x(1)^2 - x(2)^4), ...
    (-12 * x(1)^2 * x(2)^3 + 4 * x(1)^4 * x(2)^3) * exp(-x(1)^2 - x(2)^4);
    (-12 * x(1)^2 * x(2)^3 + 4 * x(1)^4 * x(2)^3) * exp(-x(1)^2 - x(2)^4), ...
    (-12*x(1)^3*x(2)^2 + 16*x(1)^3*x(2)^6) * exp(-x(1)^2 - x(2)^4)
];

% Αρχικό σημείο
xk = [1; 1];
X = xk'; % για αποθήκευση των x

% Μέγιστες επαναλήψεις
maxIter = 100;
tol = 1e-6;

for k = 1:maxIter
    gk = gradf(xk);
    Hk = Hessf(xk);
    
    if norm(gk) < tol
        break;
    end

    % Κατεύθυνση Newton
    dk = -Hk \ gk;

    % Exact Line Search για εύρεση gamma
    phi = @(gamma) f(xk + gamma * dk);
    gamma_k = fminbnd(phi, 0, 1); % ελαχιστοποίηση στο διάστημα [0,1]

    % Ενημέρωση
    xk = xk + gamma_k * dk;
    X = [X; xk'];
end

% Σχεδίαση επιπέδου καμπυλών και πορείας
[xgrid, ygrid] = meshgrid(linspace(-2, 2, 100), linspace(-2, 2, 100));
zgrid = arrayfun(@(x,y) f([x;y]), xgrid, ygrid);

figure;
contour(xgrid, ygrid, zgrid, 30);
hold on;
plot(X(:,1), X(:,2), 'ro-', 'LineWidth', 2);
plot(X(1,1), X(1,2), 'wo', 'MarkerSize', 8, 'LineWidth', 2); % αρχικό σημείο
title('Πορεία Newton με Exact Line Search (αρχή στο (1,1))');
xlabel('x'); ylabel('y');
grid on;
