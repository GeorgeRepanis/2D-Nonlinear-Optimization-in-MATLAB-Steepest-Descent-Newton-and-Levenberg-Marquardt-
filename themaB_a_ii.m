clc; clear; close all;

%% Συνάρτηση και gradient
f = @(x,y) x.^3 .* exp(-x.^2 - y.^4);

gradf = @(x,y) [ ...
    x.^2 .* (3 - 2*x.^2) .* exp(-x.^2 - y.^4);      % df/dx
   -4*x.^3 .* y.^3 .* exp(-x.^2 - y.^4)            % df/dy
];

%% Παράμετροι Steepest Descent
gamma   = 0.2;                % σταθερό βήμα
maxIter = 30;                % αριθμός επαναλήψεων
xk      = [-1; -1];           % αρχικό σημείο x0 = (-1,-1)

fvals   = zeros(maxIter+1,1);
fvals(1)= f(xk(1),xk(2));

for k = 1:maxIter
    gk = gradf(xk(1), xk(2));
    xk = xk - gamma * gk;     % Μέθοδος Μέγιστης Καθόδου
    fvals(k+1) = f(xk(1), xk(2));
end

%% "Πραγματικό" τοπικό ελάχιστο (απ' την ανάλυση)
f_true_min = -0.41;

%% Σχεδίαση διαγράμματος f(x_k) vs k
kvec = 0:maxIter;

figure('Color',[0 0 0]);      % μαύρο background, όπως στο παράδειγμα
axes('Color',[0 0 0], ...
     'XColor',[1 1 1], ...
     'YColor',[1 1 1], ...
     'FontSize',12); hold on; grid on;

plot(kvec, fvals,'-o','LineWidth',1.5,'MarkerSize',4, ...
     'MarkerFaceColor','none','Color',[0.3 0.6 1]);

yline(f_true_min,'--','f_{true min} \approx -0.41', ...
      'Color',[0.8 0.8 0.8],'LineWidth',1.2, ...
      'LabelHorizontalAlignment','left', ...
      'LabelVerticalAlignment','bottom');

xlabel('Επαναλήψεις k','Color','w','FontSize',14);
ylabel('f(x_k)','Color','w','FontSize',14);
title('f(x_k) για Steepest Descent με x_0 = (-1,-1) και \gamma = 0.2', ...
      'Color','w','FontSize',14);

legend({'f(x_k)','Πραγματικό τοπικό ελάχιστο'}, ...
       'TextColor','w','Location','northeast');

xlim([0 maxIter]);
