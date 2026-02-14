%% Μέθοδος Newton για f(x,y) = x^3 * exp(-x^2 - y^4)
% Αρχικό σημείο: (-1, -1), σταθερό βήμα gamma = 0.5
% Στόχος: να δείξουμε ότι συγκλίνει προς f ~ 0 και ΟΧΙ στο τοπικό ελάχιστο f_min ~ -0.41

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

% Πραγματικό τοπικό ελάχιστο (θεωρητικό, y=0, x = -sqrt(3/2))
x_star = [-sqrt(3/2); 0];
f_true_min = f(x_star);   % ~ -0.41

% Παράμετροι Newton
gamma   = 0.5;    % σταθερό βήμα, 0<gamma<1
maxIter = 20;     % λίγες επαναλήψεις για καθαρό γράφημα
tol_g   = 1e-6;   % όριο gradient
tol_f   = 1e-4;   % σταματάμε όταν |f| ~ 0 (φαίνεται η σύγκλιση στο 0)

% Αρχικό σημείο
xk = [-1; -1];

% Ιστορία
Xhist = xk;
Fhist = f(xk);

fprintf('k = %2d, x = (% .6f, % .6f), f = % .6e, ||grad f|| = %.3e\n', ...
        0, xk(1), xk(2), Fhist(end), norm(gradf(xk)));

for k = 1:maxIter

    gk = gradf(xk);
    if norm(gk) < tol_g
        fprintf('Στάση στη διαδοχή k = %d (||grad f|| < tol_g)\n', k-1);
        break;
    end

    Hk = Hf(xk);
    if rcond(Hk) < 1e-12
        fprintf('Στάση στη διαδοχή k = %d (Hessian μη αντιστρέψιμος)\n', k-1);
        break;
    end

    pk = - Hk \ gk;
    x_new = xk + gamma * pk;

    % αν η f έχει ήδη σχεδόν μηδενιστεί, σταματάμε (για να φαίνεται η σύγκλιση στο 0)
    if abs(f(x_new)) < tol_f
        xk = x_new;
        Xhist(:,end+1) = xk;
        Fhist(end+1)   = f(xk);
        fprintf('Στάση στη διαδοχή k = %d (|f(x_k)| < tol_f)\n', k);
        break;
    end

    xk = x_new;
    Xhist(:,end+1) = xk;
    Fhist(end+1)   = f(xk);

    fprintf('k = %2d, x = (% .6f, % .6f), f = % .6e, ||grad f|| = %.3e\n', ...
            k, xk(1), xk(2), Fhist(end), norm(gk));
end

%% Διάγραμμα σύγκλισης f(x_k) vs k με γραμμή στο τοπικό ελάχιστο

k_axis = 0:length(Fhist)-1;

figure;
plot(k_axis, Fhist, 'o-','LineWidth',1.2); hold on;
yline(f_true_min, '--', sprintf('f_{true min} = %.2f', f_true_min), ...
      'LabelHorizontalAlignment','left');

xlabel('Επαναλήψεις k');
ylabel('f(x_k)');
title('Σύγκλιση Newton με x_0 = (-1,-1) και \gamma = 0.5');
grid on;
