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
alpha = 1e-4;   % 0 < alpha < 1
beta  = 0.5;    % 0 < beta < 1  (συντελεστής συστολής)
s     = 1;      % αρχικό βήμα

%% Newton + Armijo από x0 = (-1,-1)
xk = [-1; -1];      % αρχικό σημείο
maxIter = 50;
tol = 1e-6;

xs = zeros(2, maxIter+1);   % για αποθήκευση της πορείας
xs(:,1) = xk;

numIter = 0;

for k = 1:maxIter
    gk = gradf(xk);

    if norm(gk) < tol
        numIter = k-1;  % δεν έγινε νέο βήμα
        break;
    end

    Hk = Hessf(xk);

    % ---- Διεύθυνση Newton με safeguard ----
    if rcond(Hk) < 1e-12
        % κακός Hessian -> γυρνάμε σε steepest descent
        dk = -gk;
    else
        dk = -Hk \ gk;   % κλασική διεύθυνση Newton
        if gk.'*dk >= 0
            % δεν είναι κατεύθυνση καθόδου -> χρησιμοποιώ -grad
            dk = -gk;
        end
    end

    % ---- Κανόνας Armijo (backtracking) ----
    gamma = s;
    fk = f(xk);
    while f(xk + gamma*dk) > fk + alpha*gamma*(gk.'*dk)
        gamma = beta * gamma;
        if gamma < 1e-10
            % δεν βρίσκουμε αποδεκτό βήμα
            break;
        end
    end

    % ενημέρωση σημείου
    xk = xk + gamma*dk;
    xs(:,k+1) = xk;
    numIter = k;
end

if numIter == 0
    numIter = 1; % μόνο το αρχικό σημείο
end

%% Διάγραμμα πορείας Newton + Armijo
% πλέγμα για τα ισοϋψή
[xGrid, yGrid] = meshgrid(linspace(-2, 2, 400), linspace(-2, 2, 400));
F = (xGrid.^3) .* exp(-xGrid.^2 - yGrid.^4);

figure;
contour(xGrid, yGrid, F, 30); hold on;
colormap turbo;
colorbar;

% πορεία
plot(xs(1,1:numIter+1), xs(2,1:numIter+1), 'r.-', 'LineWidth', 2, 'MarkerSize', 12);

% αρχικό και τελικό σημείο
plot(xs(1,1), xs(2,1), 'wo', 'MarkerSize', 8, 'LineWidth', 2);        % start
plot(xs(1,numIter+1), xs(2,numIter+1), 'ro', 'MarkerSize', 8, 'LineWidth', 2); % end

xlabel('x');
ylabel('y');
title('Πορεία Newton με κανόνα Armijo (αρχή στο (-1,-1))');
grid on;
axis equal;
