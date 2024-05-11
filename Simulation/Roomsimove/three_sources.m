
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT WHICH PERFORMS THE SIMULATION FOR 3 SOURCES AND CALCULATES THE
% SOUND PRESSURE.
% Author: sergiozc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

addpath(genpath('stft_library'))

%% Sources definition
% FIRST SOURCE
[s1,fs] = audioread('sources/scarface_alpacino.wav');
% Position (in meters)
x1 = 3;
y1 = 2.1;
z1 = 1.65; %eg: another human height


% SECOND SOURCE
[s2,fs] = audioread('sources/every_man_De_Niro.wav');
s2 = s2(1:length(s1));  % Same length as source 1
% Position (in meters)
x2 = 1;
y2 = 3.5;
z2 = 1.8; %eg: human height

% THIRD SOURCE
[s3,fs] = audioread('sources/eastwood_lawyers.wav');
% Position (in meters)
s3 = s3(1:length(s1)); % Same length as source 1
x3 = 4;
y3 = 1.5;
z3 = 1.7; %eg: another human height

%% Sources visualization
% TIME VISUALIZATION
% Time vector
dura1 = length(s1) / fs;
time1 = linspace(0, dura1, length(s1));
dura2 = length(s2) / fs;
time2 = linspace(0, dura2, length(s2));
dura3 = length(s3) / fs;
time3 = linspace(0, dura3, length(s3));
figure(1);
plot(time1, s1);
hold on;
plot(time2, s2);
plot(time3, s3);
hold off;
xlabel('Time (s)');
ylabel('Amplitude');
title('Sources in time domain');
legend('Source 1', 'Source 2', 'Source 3');

% FREQUENCY DOMAIN
s1_f = fft(s1);
s2_f = fft(s2);
s3_f = fft(s3);
freq = linspace(0, fs, length(s1_f));
figure(2);
plot(freq(1:length(s1_f)/2), abs(s1_f(1:length(s1_f)/2)));
hold on;
plot(freq(1:length(s1_f)/2), abs(s2_f(1:length(s1_f)/2)));
plot(freq(1:length(s1_f)/2), abs(s3_f(1:length(s1_f)/2)));
hold off;
grid on;
title('Sources in frequency domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
legend('Source 1', 'Source 2', 'Source 3');

% SPATIAL VISUALIZATION
% Room dimensions (same as room_sensor_config.txt)
room_size = [5, 4, 2.6];
% Microphones positions
pos_mic = [
    1.0000, 1.0000, 1.3420;
    1.0148, 1.0015, 1.3393;
    1.0202, 1.0040, 1.3366;
    1.0238, 1.0072, 1.3339;
    1.0260, 1.0108, 1.3312;
    1.0272, 1.0146, 1.3285;
    1.0276, 1.0184, 1.3257;
    1.0271, 1.0223, 1.3230;
    1.0260, 1.0260, 1.3203;
    1.0242, 1.0295, 1.3176;
    1.0218, 1.0326, 1.3149;
    1.0189, 1.0354, 1.3122;
    1.0157, 1.0378, 1.3095;
    1.0120, 1.0397, 1.3068;
    1.0082, 1.0410, 1.3041;
    1.0041, 1.0418, 1.3014;
    1.0000, 1.0420, 1.2986;
    0.9959, 1.0416, 1.2959;
    0.9919, 1.0407, 1.2932;
    0.9881, 1.0392, 1.2905;
    0.9846, 1.0371, 1.2878;
    0.9815, 1.0346, 1.2851;
    0.9788, 1.0317, 1.2824;
    0.9767, 1.0284, 1.2797;
    0.9752, 1.0248, 1.2770;
    0.9743, 1.0211, 1.2743;
    0.9743, 1.0172, 1.2715;
    0.9752, 1.0133, 1.2688;
    0.9771, 1.0095, 1.2661;
    0.9803, 1.0060, 1.2634;
    0.9854, 1.0029, 1.2607;
    1.0000, 1.0000, 1.2580;];
% Separating microphones positions
x_mic = pos_mic(:, 1);
y_mic = pos_mic(:, 2);
z_mic = pos_mic(:, 3);

% Mostrar las posiciones de los micrófonos
figure;
scatter3(x_mic, y_mic, z_mic, 'filled');
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('Sensors geometry');
grid on;

% Visualización espacial
figure;
scatter3(x1, y1, z1, 'filled');
hold on
scatter3(x2, y2, z2, 'filled');
scatter3(x3, y3, z3, 'filled');
scatter3(x_mic, y_mic, z_mic, 'filled', 'MarkerFaceColor', 'r'); % Agregar micrófonos en rojo
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
xlim([0 room_size(1)]);
ylim([0 room_size(2)]);
zlim([0 room_size(3)]);
title('Sources and Microphones positions');
legend('Source 1', 'Source 2', 'Source 3', 'Microphones');

%% Simulation
Nmic = 32; % Number of microphones

H1 = roomsimove_single('room_sensor_config.txt',[x1; y1; z1]);
H2 = roomsimove_single('room_sensor_config.txt',[x2; y2; z2]);
H3 = roomsimove_single('room_sensor_config.txt',[x3; y3; z3]);

% Received signal at the qth microphone
y = fftfilt(H1,s1) + fftfilt(H2,s2) + fftfilt(H3,s3);

%% Sound pressure
winlen = uint32(8);
% winlen = 256; % It means 256 ms
hop = 0.5;  % Overlap. Default is 50%, or 0.5
nfft = 128; % Default is same length as winlen

% Transformada de fourier para cada ventan
stftObj = STFTClass(fs, winlen, hop, nfft);

% Perform the STFT on y
T = 500; % Number of time frames

% Inicializar P como una matriz tridimensional
Nfreq = stftObj.pos_freq; % Number of frequencies
P = zeros(Nfreq, Nmic, T);

for n = 1:Nmic
    % STFT in each microphone signal
    P(:, n, :) = stftObj.stft(y(:, n), T);
end


%% Saving data

% Saving sensors positions (cartesian)
save('../PSD-algorithm/data/pos_mic.mat', 'pos_mic');

% Saving sources positions (cartesian)
pos_sources = [x1, y1, z1; x2, y2, z2; x3, y3, z3];
save('../PSD-algorithm/data/pos_sources.mat', 'pos_sources');

% Saving frequency array (to create a tensor in python)
freq_array = stftObj.freqArray; % Number of frequencies
save('../PSD-algorithm/data/freq.mat', 'freq_array');

% Saving sound pressure tensor
save('../PSD-algorithm/data/sound_pressure.mat', 'P');


%% Recorded signals

% Each microphone
mic1 = y(:, 1);
mic2 = y(:, 2);
mic3 = y(:, 32);

%soundsc(mic1, fs);
%soundsc(mic2, fs);
%soundsc(mic3, fs);

figure(3);
plot(time1, mic1);
hold on;
plot(time1, mic2);
plot(time1, mic3);
hold off;
xlabel('Time (s)');
ylabel('Amplitude');
title('Recorded signals (mic #1, #2 and #32)');
legend('Mic 1', 'Mic 2', 'Mic 32');

