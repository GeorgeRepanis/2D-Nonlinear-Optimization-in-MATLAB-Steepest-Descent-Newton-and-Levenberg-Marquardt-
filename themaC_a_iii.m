%% Μέθοδος Newton για f(x,y) = x^3 * exp(-x^2 - y^4)
% Αρχικό σημείο: (1, 1), σταθερό βήμα gamma = 0.5

clear; clc; close all;

% Συνάρτηση f
f = @(x) x(1).^3 .* exp(-x(1).^2 - x(2).^4);

% Gradient της f
gradf = @(x) [ ...
    x(1).^2 .* (3 - 2*x(1).^2) .* exp(-x(1).^2 - x(2).^4); ...
   -4*x(1).^3 .* x(2).^3 .* exp(-x(1).^2 - x(2).^4) ];

% Hessian της f
Hf = @(x) [ ...
    (4*x(1).^5 - 14*x(1).^3 + 6*x(1)) .* exp(-x(1).^2 - x(2).^4), ...
    x(1).^2 .* x(2).^3 .* (8*x(1).^2 - 12) .* exp(-x(1).^2 - x(2).^4); ...
    x(1).^2 .* x(2).^3 .* (8*x(1).^2 - 12) .* exp(-x(1).^2 - x(2).^4), ...
    x(1).^3 .* x(2).^2 .* (16*x(2).^4 - 12) .* exp(-x(1).^2 - x(2).^4) ...
];

% Παράμετροι
gamma   = 0.5;    % σταθερό βήμα Newton, 0 < gamma < 1
maxIter = 7;
tol     = 1e-6;

% Αρχικό σημείο
xk = [1; 1];

% Ιστορία
Xhist = xk;
Fhist = f(xk);

gk = gradf(xk);
fprintf('k = %2d, x = (% .6f, % .6f), f = % .6f, ||grad f|| = %.3e\n', ...
        0, xk(1), xk(2), Fhist(end), norm(gk));

for k = 1:maxIter

    gk = gradf(xk);

    % Κριτήριο τερματισμού
    if norm(gk) < tol
        fprintf('Στάση στη διαδοχή k = %d (||grad f|| < tol)\n', k-1);
        break;
    end

    Hk = Hf(xk);

    % Έλεγχος αντιστρεψιμότητας Hessian
    if rcond(Hk) < 1e-12
        fprintf('Στάση στη διαδοχή k = %d (Hessian μη αντιστρέψιμος στο x_k)\n', k-1);
        break;
    end

    % Βήμα Newton
    pk = - Hk \ gk;
    xk = xk + gamma * pk;

    Xhist(:,end+1) = xk;
    Fhist(end+1)   = f(xk);

    fprintf('k = %2d, x = (% .6f, % .6f), f = % .6f, ||grad f|| = %.3e\n', ...
            k, xk(1), xk(2), Fhist(end), norm(gk));
end

%% Διάγραμμα σύγκλισης f(x_k) vs k

k_axis = 0:length(Fhist)-1;

figure;
plot(k_axis, Fhist, 'o-','LineWidth',1.2);
xlabel('Επαναλήψεις k');
ylabel('f(x_k)');
title('Σύγκλιση Newton με x_0 = (1,1) και \gamma = 0.5');
grid on;
