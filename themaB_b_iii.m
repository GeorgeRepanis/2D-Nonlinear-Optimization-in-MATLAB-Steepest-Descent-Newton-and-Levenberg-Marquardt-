%% Steepest Descent με exact line search για f(x,y) = x^3 * exp(-x^2 - y^4)
% Αρχικό σημείο: (1, 1)
clear; clc; close all;

% Συνάρτηση f
f = @(x) x(1).^3 .* exp(-x(1).^2 - x(2).^4);

% Gradient της f
gradf = @(x) [ ...
    x(1).^2 .* (3 - 2*x(1).^2) .* exp(-x(1).^2 - x(2).^4); ...
   -4*x(1).^3 .* x(2).^3 .* exp(-x(1).^2 - x(2).^4) ];

% Παράμετροι
maxIter = 50;
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

    dk = -gk;   % διεύθυνση μέγιστης καθόδου

    % ---- Exact line search: γ_k = argmin f(x_k + γ d_k) ----
    phi = @(gamma) f(xk + gamma * dk);
    gamma_k = fminbnd(phi, 0, 5);   % διάστημα αναζήτησης [0,5]
    % --------------------------------------------------------

    % Νέο σημείο
    xk = xk + gamma_k * dk;

    Xhist(:,end+1) = xk;
    Fhist(end+1)   = f(xk);

    fprintf('k = %3d, γ_k = %.4f, x = (% .6f, % .6f), f = % .6f, ||grad f|| = %.3e\n', ...
            k, gamma_k, xk(1), xk(2), Fhist(end), norm(gk));
end

%% Διάγραμμα σύγκλισης f(x_k) vs k

k_axis = 0:length(Fhist)-1;

figure;
plot(k_axis, Fhist, 'o-','LineWidth',1.2);
xlabel('Επαναλήψεις k');
ylabel('f(x_k)');
title('Σύγκλιση f(x_k) με Steepest Descent (exact line search), x_0 = (1,1)');
grid on;
