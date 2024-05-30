
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT WHICH DEFINES THE CORRESPONDING POSITIONS FOR SINGLE SENSORS IN A 
% HIGH ORDER MICROPHONE
% Author: sergiozc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear all;
close all;

%% Creating a custom geometry
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
title('Microphone positions within the sphere');
grid on;

% Print the positions
disp('Microphones positions:');
for i = 1:num_mic
    fprintf('Sensor %d: x = %.4f m, y = %.4f m, z = %.4f m\n', i, x(i), y(i), z(i));
end

%% Eigenmike geometry
addpath(genpath('SH_library'))

hom = SHTools.getEigenmike();
cart_eigen = hom.cart;

% Center of the sphere
center = [1, 1, 1];
% Shift the geometry 
cart_eigen_shift = cart_eigen + center;

x_eigen = cart_eigen_shift(:, 1);
y_eigen = cart_eigen_shift(:, 2);
z_eigen = cart_eigen_shift(:, 3);
% Spatial visualization
figure;
scatter3(x_eigen, y_eigen, z_eigen, 'filled');
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('Eigenmike geometry');
grid on;

% Print the position (proper format to "room_sensor_config.txt")
for i = 1:num_mic
    fprintf('sp%d     %.4f  %.4f  %.4f\n', i, ...
        cart_eigen_shift(i, 1), cart_eigen_shift(i, 2), cart_eigen_shift(i, 3));
end

