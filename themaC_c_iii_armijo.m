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
alpha = 1e-4;
beta  = 0.5;
s     = 1;

%% Newton + Armijo από x0 = (1,1)
xk = [1; 1];
maxIter = 50;
tol = 1e-6;

xs = zeros(2, maxIter+1);
xs(:,1) = xk;
numIter = 0;

for k = 1:maxIter
    gk = gradf(xk);
    if norm(gk) < tol
        numIter = k-1;
        break;
    end

    Hk = Hessf(xk);

    % διεύθυνση Newton με safeguard
    if rcond(Hk) < 1e-12
        dk = -gk;                 % fallback σε steepest descent
    else
        dk = -Hk \ gk;
        if gk.'*dk >= 0           % αν δεν είναι καθόδου
            dk = -gk;
        end
    end

    % Armijo backtracking
    gamma = s;
    fk = f(xk);
    while f(xk + gamma*dk) > fk + alpha*gamma*(gk.'*dk)
        gamma = beta * gamma;
        if gamma < 1e-10
            break;
        end
    end

    % ενημέρωση
    xk = xk + gamma*dk;
    xs(:,k+1) = xk;
    numIter = k;
end

if numIter == 0
    numIter = 1;
end

%% Πλέγμα και ισοϋψείς για f
[xGrid, yGrid] = meshgrid(linspace(-2, 2, 400), linspace(-2, 2, 400));
F = (xGrid.^3) .* exp(-xGrid.^2 - yGrid.^4);

%% Διάγραμμα πορείας
figure;
contour(xGrid, yGrid, F, 30); hold on;
colormap turbo;
colorbar;

plot(xs(1,1:numIter+1), xs(2,1:numIter+1), 'r.-', 'LineWidth', 2, 'MarkerSize', 12);

% αρχικό (λευκό) και τελικό (κόκκινο) σημείο
plot(xs(1,1), xs(2,1), 'wo', 'MarkerSize', 8, 'LineWidth', 2);
plot(xs(1,numIter+1), xs(2,numIter+1), 'ro', 'MarkerSize', 8, 'LineWidth', 2);

xlabel('x');
ylabel('y');
title('Πορεία Newton με κανόνα Armijo (αρχή στο (1,1))');
grid on;
axis equal;
