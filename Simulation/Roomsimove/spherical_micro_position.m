
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT WHICH DEFINES THE CORRESPONDING POSITIONS FOR SINGLE SENSORS IN A 
% HIGH ORDER MICROPHONE (E.G. EIGENMIKE)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



clc;
clear all;
close all;

% Definir el número de micrófonos
num_mic = 32;

% Radio de la esfera (en metros)
radio_esfera = 0.042;  % 4.2 cm en metros

% Coordenadas del centro de la esfera
centro_esfera = [1, 1, 1.3];

% Calcular las posiciones de los micrófonos en coordenadas esféricas
theta = pi * (0:num_mic-1) / num_mic; % Ángulo polar
phi = acos(1 - 2 * (0:num_mic-1) / (num_mic - 1)); % Ángulo azimutal
r = radio_esfera * ones(1, num_mic); % Radio constante

% Convertir coordenadas esféricas a cartesianas
x = r .* sin(phi) .* cos(theta) + centro_esfera(1);
y = r .* sin(phi) .* sin(theta) + centro_esfera(2);
z = r .* cos(phi) + centro_esfera(3);


% Mostrar las posiciones de los micrófonos
figure;
scatter3(x, y, z, 'filled');
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('Posiciones de los micrófonos en la habitación');
grid on;

disp('Posiciones de los micrófonos:');
for i = 1:num_mic
    fprintf('Sensor %d: x = %.4f m, y = %.4f m, z = %.4f m\n', i, x(i), y(i), z(i));
end
