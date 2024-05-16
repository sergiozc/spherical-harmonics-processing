
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT WHICH DEFINES THE CORRESPONDING POSITIONS FOR SINGLE SENSORS IN A 
% HIGH ORDER MICROPHONE
% Author: sergiozc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear all;
close all;

% Number of sensors (mics)
num_mic = 32;
% radius
radio = 0.042;
% sphere center
center = [1, 1, 1.3];

% Micro position in spherical uniform coordinates
theta = acos(1 - 2 * (1:num_mic) / (num_mic + 1)); % Ángulo polar
phi = pi * (1 + sqrt(5)) * (1:num_mic); % Ángulo azimutal, distribución uniforme

% spherical to cartesian coordinates
x = radio * sin(theta) .* cos(phi) + center(1);
y = radio * sin(theta) .* sin(phi) + center(2);
z = radio * cos(theta) + center(3);

% Spatial visualization
figure;
scatter3(x, y, z, 'filled');
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('Posiciones de los micrófonos en la esfera');
grid on;

% Print the positions
disp('Microphones positions:');
for i = 1:num_mic
    fprintf('Sensor %d: x = %.4f m, y = %.4f m, z = %.4f m\n', i, x(i), y(i), z(i));
end
