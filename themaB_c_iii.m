%% Steepest Descent με κανόνα Armijo για f(x,y) = x^3 * exp(-x^2 - y^4)
% Αρχικό σημείο: (1, 1)

clear; clc; close all;

% Συνάρτηση f
f = @(x) x(1).^3 .* exp(-x(1).^2 - x(2).^4);

% Gradient της f
gradf = @(x) [ ...
    x(1).^2 .* (3 - 2*x(1).^2) .* exp(-x(1).^2 - x(2).^4); ...
   -4*x(1).^3 .* x(2).^3 .* exp(-x(1).^2 - x(2).^4) ];

% Παράμετροι Armijo
sigma   = 1e-4;   % 0 < sigma < 1
beta    = 0.5;    % παράγοντας μείωσης (0 < beta < 1)
gamma0  = 1.0;    % αρχικό βήμα για backtracking

maxIter = 3;
tol     = 1e-6;

% Αρχικό σημείο
xk = [1; 1];

% Ιστορία
Xhist = xk;
Fhist = f(xk);

fprintf('k = %3d, x = (% .6f, % .6f), f = % .6f, ||grad f|| = %.3e\n', ...
        0, xk(1), xk(2), Fhist(end), norm(gradf(xk)));

for k = 1:maxIter
    gk = gradf(xk);

    % Κριτήριο τερματισμού
    if norm(gk) < tol
        fprintf('Στάση στη διαδοχή k = %d (||grad f|| < tol)\n', k-1);
        break;
    end

    dk = -gk;          % διεύθυνση μέγιστης καθόδου
    gamma = gamma0;    % αρχικό βήμα για Armijo

    % Backtracking Armijo
    while f(xk + gamma*dk) > f(xk) + sigma * gamma * (gk.' * dk)
        gamma = beta * gamma;
    end

    % Ενημέρωση σημείου
    xk = xk + gamma * dk;

    Xhist(:,end+1) = xk;
    Fhist(end+1)   = f(xk);

    fprintf('k = %3d, γ_k = %.4f, x = (% .6f, % .6f), f = % .6f, ||grad f|| = %.3e\n', ...
            k, gamma, xk(1), xk(2), Fhist(end), norm(gk));
end

%% Διάγραμμα σύγκλισης f(x_k) vs k

k_axis = 0:length(Fhist)-1;

figure;
plot(k_axis, Fhist, 'o-','LineWidth',1.2);
xlabel('Επαναλήψεις k');
ylabel('f(x_k)');
title('Σύγκλιση f(x_k) με Steepest Descent (Armijo), x_0 = (1,1)');
grid on;
