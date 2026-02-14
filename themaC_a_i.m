%% Μέθοδος Newton για f(x,y) = x^3 * exp(-x^2 - y^4)
% Αρχικό σημείο: (0, 0), σταθερό βήμα gamma = 1

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
gamma   = 0.5;     % σταθερό βήμα Newton
maxIter = 50;
tol     = 1e-6;

% Αρχικό σημείο
xk = [0; 0];

% Ιστορία
Xhist = xk;
Fhist = f(xk);

gk = gradf(xk);
fprintf('k = %2d, x = (% .6f, % .6f), f = % .6f, ||grad f|| = %.3e\n', ...
        0, xk(1), xk(2), Fhist(end), norm(gk));

for k = 1:maxIter

    gk = gradf(xk);

    % Αν το gradient είναι ήδη ~0, ο Newton "νομίζει" ότι τελειώσαμε
    if norm(gk) < tol
        fprintf('Στάση στη διαδοχή k = %d (||grad f|| < tol)\n', k-1);
        break;
    end

    Hk = Hf(xk);

    % Προσπάθεια επίλυσης Hk * p = -gk
    % (εδώ στο (0,0) ο Hk είναι μηδενικός, άρα singular)
    if rcond(Hk) < 1e-12
        fprintf('Στάση στη διαδοχή k = %d (Hessian μη αντιστρέψιμος στο x_k)\n', k-1);
        break;
    end

    pk = - Hk \ gk;       % διεύθυνση Newton
    xk = xk + gamma * pk; % ενημέρωση

    Xhist(:,end+1) = xk;
    Fhist(end+1)   = f(xk);

    fprintf('k = %2d, x = (% .6f, % .6f), f = % .6f, ||grad f|| = %.3e\n', ...
            k, xk(1), xk(2), Fhist(end), norm(gk));
end

%% Διάγραμμα f(x_k) vs k (θα είναι σταθερό f = 0)

k_axis = 0:length(Fhist)-1;

figure;
plot(k_axis, Fhist, 'o-','LineWidth',1.2);
xlabel('Επαναλήψεις k');
ylabel('f(x_k)');
title('Σύγκλιση Newton με x_0 = (0,0) και \gamma = 0.5 (αποτυχία)');
grid on;
