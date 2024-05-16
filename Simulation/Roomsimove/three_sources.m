
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT WHICH PERFORMS THE SIMULATION FOR 3 SOURCES AND CALCULATES THE
% SOUND PRESSURE.
% Author: sergiozc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

addpath(genpath('stft_library'))

%% Sources definition (fs in "room_sensor_config.txt too)

% FIRST SOURCE
%[s1,fs] = audioread('sources/scarface_alpacino.wav');
freq1 = 440; fs = 16000; duration = 5; % Hz, samples/s, s
s1 = sin_source_generation(freq1, duration, fs);

% Position (in meters)
x1 = 3;
y1 = 2.1;
z1 = 1.65; %eg: another human height


% SECOND SOURCE
%[s2,fs] = audioread('sources/every_man_De_Niro.wav');
%s2 = s2(1:length(s1));  % Same length as source 1

freq2 = 900; fs = 16000; duration = 5; % Hz, samples/s, s
s2 = sin_source_generation(freq2, duration, fs);

% Position (in meters)
x2 = 1;
y2 = 3.5;
z2 = 1.8; %eg: human height

% THIRD SOURCE
%[s3,fs] = audioread('sources/eastwood_lawyers.wav');
%s3 = s3(1:length(s1)); % Same length as source 1

freq3 = 2500; fs = 16000; duration = 5; % Hz, samples/s, s
s3 = sin_source_generation(freq3, duration, fs);

% Position (in meters)
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
figure;
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
figure;
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
    0.9894, 0.9903, 1.3395;
    1.0018, 1.0200, 1.3369;
    1.0147, 0.9808, 1.3344;
    0.9730, 1.0048, 1.3318;
    1.0254, 1.0162, 1.3293;
    0.9916, 0.9687, 1.3267;
    0.9842, 1.0305, 1.3242;
    1.0338, 0.9877, 1.3216;
    0.9654, 0.9857, 1.3191;
    1.0164, 1.0350, 1.3165;
    1.0119, 0.9622, 1.3140;
    0.9650, 1.0203, 1.3115;
    1.0401, 1.0088, 1.3089;
    0.9761, 0.9660, 1.3064;
    0.9946, 1.0415, 1.3038;
    1.0321, 0.9729, 1.3013;
    0.9581, 0.9983, 1.2987;
    1.0296, 1.0295, 1.2962;
    0.9981, 0.9585, 1.2936;
    0.9737, 1.0315, 1.2911;
    1.0400, 0.9946, 1.2885;
    0.9675, 0.9774, 1.2860;
    1.0085, 1.0377, 1.2835;
    1.0186, 0.9675, 1.2809;
    0.9657, 1.0109, 1.2784;
    1.0312, 1.0144, 1.2758;
    0.9875, 0.9701, 1.2733;
    0.9898, 1.0283, 1.2707;
    1.0243, 0.9872, 1.2682;
    0.9766, 0.9938, 1.2656;
    1.0108, 1.0169, 1.2631;
    1.0024, 0.9858, 1.2605;
];
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

% Fourier transform for each window
stftObj = STFTClass(fs, winlen, hop, nfft);

% Perform the STFT on y
T = 500; % Number of time frames

% Sound pressure as a tensor
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

figure;
plot(time1, mic1);
hold on;
plot(time1, mic2);
plot(time1, mic3);
hold off;
xlabel('Time (s)');
ylabel('Amplitude');
title('Recorded signals (mic #1, #2 and #32)');
legend('Mic 1', 'Mic 2', 'Mic 32');

%% Sources' PSD representation
% Calcular la STFT
[S, F, T] = spectrogram(s1, winlen, winlen/2, [], fs);

magnitud = abs(S);
PSD = magnitud.^2;

figure;
imagesc(T, F, 10*log10(PSD));
colorbar;
axis xy;
caxis([-60, 0])
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('PSD source 1');

[S, F, T] = spectrogram(s2, winlen, winlen/2, [], fs);

magnitud = abs(S);
PSD = magnitud.^2;
figure;
imagesc(T, F, 10*log10(PSD));
colorbar;
axis xy;
caxis([-60, 0])
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('PSD source 2');

[S, F, T] = spectrogram(s3, winlen, winlen/2, [], fs);
magnitud = abs(S);
PSD = magnitud.^2;
figure;
imagesc(T, F, 10*log10(PSD));
colorbar;
axis xy;
caxis([-60, 0])
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('PSD source 3');
