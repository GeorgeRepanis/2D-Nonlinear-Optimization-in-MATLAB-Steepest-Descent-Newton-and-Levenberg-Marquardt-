% Γραφική απεικόνιση της f(x,y) = x^3 * exp(-x^2 - y^4)

% Περιοχή σχεδίασης (μπορείς να την αλλάξεις αν θέλεις)
x_min = -3;  x_max = 3;
y_min = -3;  y_max = 3;

% Δημιουργία πλέγματος
nx = 200; ny = 200;            % πυκνότητα σημείων
[x, y] = meshgrid(linspace(x_min, x_max, nx), ...
                  linspace(y_min, y_max, ny));

% Ορισμός της συνάρτησης
f = x.^3 .* exp(-x.^2 - y.^4);

%% 3D επιφάνεια
figure;
surf(x, y, f);
shading interp;
xlabel('x');
ylabel('y');
zlabel('f(x,y)');
title('3D επιφάνεια της f(x,y) = x^3 e^{-x^2 - y^4}');
grid on;
colorbar;

%% Ισοϋψείς καμπύλες (contour plot)
figure;
contourf(x, y, f, 30);     % 30 επίπεδα ισοϋψών
xlabel('x');
ylabel('y');
title('Ισοϋψείς καμπύλες της f(x,y) = x^3 e^{-x^2 - y^4}');
grid on;
colorbar;
axis equal;
