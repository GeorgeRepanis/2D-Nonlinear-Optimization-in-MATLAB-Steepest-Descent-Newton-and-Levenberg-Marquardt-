clear; clc; close all;

%% Συνάρτηση, gradient, Hessian
f = @(x) x(1).^3 .* exp(-x(1).^2 - x(2).^4);

gradf = @(x) [ ...
    (-2*x(1).^4 + 3*x(1).^2) .* exp(-x(1).^2 - x(2).^4);   % df/dx
    -4*x(1).^3 .* x(2).^3 .* exp(-x(1).^2 - x(2).^4)       % df/dy
];

Hessf = @(x) [ ...
    (4*x(1).^5 - 14*x(1).^3 + 6*x(1)) .* exp(-x(1).^2 - x(2).^4), ...
    (8*x(1).^4.*x(2).^3 - 12*x(1).^2.*x(2).^3) .* exp(-x(1).^2 - x(2).^4); ...
    (8*x(1).^4.*x(2).^3 - 12*x(1).^2.*x(2).^3) .* exp(-x(1).^2 - x(2).^4), ...
    (16*x(1).^3.*x(2).^6 - 12*x(1).^3.*x(2).^2) .* exp(-x(1).^2 - x(2).^4) ...
];

%% Παράμετροι Armijo
alpha = 1e-4;   % τυπική τιμή
beta  = 0.5;    % συστολή βήματος (0<beta<1)
s     = 1;      % αρχικό βήμα για backtracking

%% Newton + Armijo από x0 = (-1,-1)
xk = [-1; -1];          % αρχικό σημείο
maxIter = 5;
tol = 1e-6;

f_vals = zeros(maxIter+1,1);
f_vals(1) = f(xk);
xs = zeros(2,maxIter+1);
xs(:,1) = xk;

for k = 1:maxIter
    gk = gradf(xk);
    if norm(gk) < tol
        % φτάσαμε σε στάσιμο σημείο
        f_vals(k+1:end) = f(xk);
        xs(:,k+1:end) = repmat(xk,1,maxIter-k+1);
        break;
    end

    Hk = Hessf(xk);

    % --- διεύθυνση Newton (με safeguard) ---
    if rcond(Hk) < 1e-12
        % Hessian σχεδόν μη αντιστρέψιμος -> fallback σε steepest descent
        dk = -gk;
    else
        dk = -Hk \ gk;   % κλασική διεύθυνση Newton
        if gk.'*dk >= 0
            % δεν είναι κατεύθυνση καθόδου -> χρησιμοποιώ -grad
            dk = -gk;
        end
    end

    % --- Κανόνας Armijo (backtracking) ---
    gamma = s;
    fk = f(xk);
    while f(xk + gamma*dk) > fk + alpha*gamma*(gk.'*dk)
        gamma = beta*gamma;
        if gamma < 1e-10
            % δεν βρίσκουμε αποδεκτό βήμα, σταματάμε
            break;
        end
    end

    % ενημέρωση σημείου
    xk = xk + gamma*dk;

    f_vals(k+1) = f(xk);
    xs(:,k+1)   = xk;
end

% βρες πόσες επαναλήψεις όντως έγιναν
lastIter = find(f_vals~=0,1,'last');
if isempty(lastIter), lastIter = 1; end
k_vec = 0:lastIter-1;

%% Γράφημα σύγκλισης f(x_k)
figure;
plot(k_vec, f_vals(1:lastIter), '-o','LineWidth',2,'MarkerSize',6);
grid on;
xlabel('Επαναλήψεις (k)','FontSize',12);
ylabel('f(x_k)','FontSize',12);
title('Σύγκλιση της f με Newton, Armijo από το (-1,-1)','FontSize',14);
set(gca,'FontSize',11);
