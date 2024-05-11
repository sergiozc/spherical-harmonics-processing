
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT WHICH DEFINES THE CORRESPONDING POSITIONS FOR SINGLE SENSORS IN A 
% HIGH ORDER MICROPHONE
% Author: sergiozc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear all;
close all;

% Number os microphones
num_mic = 32;
% Sphere radius (meters)
radius = 0.042;
% Coordenates of the center of the sphere
center = [1, 1, 1.3];
% Calculate microphone positions in spherical coordinates
theta = pi * (0:num_mic-1) / num_mic; % Polar angle
phi = acos(1 - 2 * (0:num_mic-1) / (num_mic - 1)); % Azimut angle
r = radius * ones(1, num_mic); % Same radius
% Spherical coordinates to cartesian
x = r .* sin(phi) .* cos(theta) + center(1);
y = r .* sin(phi) .* sin(theta) + center(2);
z = r .* cos(phi) + center(3);

% Spatial Visualizaion
figure;
scatter3(x, y, z, 'filled');
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('Positions of the microphones in the room');
grid on;

% Print the positions
disp('Microphones positions:');
for i = 1:num_mic
    fprintf('Sensor %d: x = %.4f m, y = %.4f m, z = %.4f m\n', i, x(i), y(i), z(i));
end
