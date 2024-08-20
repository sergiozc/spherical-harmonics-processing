
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT WHICH PERFORMS THE SIMULATION FOR 1 SOURCE AND CALCULATES THE
% SOUND PRESSURE. WHITE NOISE HAS BEEN ADDED.
% Author: sergiozc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

addpath(genpath('stft_library'))
addpath(genpath('SH_library'))

%% Sources definition (fs in "room_sensor_config.txt too)

fs = 16000; % 16kHz

% SOURCE (voice)
[s1,fs_voice] = audioread('sources/scarface_alpacino.wav'); % VOICE
s1 = resample(s1, fs, fs_voice); % Resample to conver fs to 16kHz % VOICE
% Position (in meters)
x1 = 4.9;
y1 = 0.1;
z1 = 1.65; %eg: human height

%% Source visualization
% FREQUENCY DOMAIN
s1_f = fft(s1);
freq = linspace(0, fs, length(s1_f));
figure;
plot(freq(1:length(s1_f)/2), abs(s1_f(1:length(s1_f)/2)));
grid on;
title('Source in frequency domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

% SPATIAL VISUALIZATION
% Room dimensions (same as room_sensor_config.txt)
room_size = [5, 4, 2.6];

cart_zylia = [
    0.0, 0.0, 49.0; % 1
    32.7, 0.1, 36.5; % 2
    -16.4, 28.3, 36.5; % 3
    -16.3, -28.3, 36.5; % 4
    6.3, -45.8, 16.3; % 5
    36.6, -28.2, 16.3; % 6
    36.5, 28.4, 16.3; % 7
    6.2, 45.8, 16.3; % 8
    -42.8, 17.4, 16.3; % 9
    -42.7, -17.6, 16.3; % 10
    -36.5, -28.4, -16.3; % 11
    -6.2, -45.8, -16.3; % 12
    42.8, -17.4, -16.3; % 13
    42.7, 17.6, -16.3; % 14
    -6.3, 45.8, -16.3; % 15
    -36.6, 28.2, -16.3; % 16
    -32.7, -0.1, -36.5; % 17
    16.4, -28.3, -36.5; % 18
    16.3, 28.3, -36.5 % 19
] ./ 1000;

% Center of the sphere
center = [1, 1, 1];
% Shift the geometry 
pos_mic = cart_zylia + center;

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
scatter3(x_mic, y_mic, z_mic, 'filled', 'MarkerFaceColor', 'r'); % Agregar micrófonos en rojo
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
xlim([0 room_size(1)]);
ylim([0 room_size(2)]);
zlim([0 room_size(3)]);
title('Source and Microphones positions');
legend('Source', 'Microphones');

%% Simulation
Nmic = 19; % Number of microphones

H1 = roomsimove_single('room_sensor_config_zylia.txt',[x1; y1; z1]);

% Received signal at qth microphone
y = fftfilt(H1,s1);

%% Sound pressure
winlen = uint32(128);
% window(s) = window(samples) / Fs
hop = 0.5;  % Overlap. Default is 50%, or 0.5
nfft = 128; % Default is same length as winlen

% Fourier transform for each window
stftObj = STFTClass(fs, winlen, hop, nfft);

T = 1300; % Number of time frames
T_vector = linspace(1,T,T); % To represent time

% Sound pressure as a tensor
Nfreq = stftObj.pos_freq; % Number of frequencies
P = zeros(Nfreq, Nmic, T);
y_noise = zeros(size(y));
for n = 1:Nmic
    % Adding white noise
    y_noise(:,n) = y(:,n) + 0.1 * randn(size(y(:,n)));
    % STFT in each microphone signal
    P(:, n, :) = stftObj.stft(y_noise(:, n), T);
end

freq_array = stftObj.freqArray; % Number of frequencies

% Spectrogram for sound pressure (mic #1)
P_psd_1 = abs(squeeze(P(:, 1, :)));
P_psd_1 = 10*log10(P_psd_1);
figure;
imagesc(T_vector, freq_array, P_psd_1);
axis xy;
colorbar_handle = colorbar; 
xlabel('Timeframes');
ylabel('Frequency (Hz)');
title('PSD of sound pressure from mic #1');
ylabel(colorbar_handle, 'PSD(dB/Hz)');
colormap('hot');
caxis([-70,0]);
%% Saving data

% Saving sensors positions (cartesian)
save('../../Experiment/SH_MVDR/input_data/pos_mic.mat', 'pos_mic');

% Saving sources positions (cartesian)
pos_sources = [x1, y1, z1];
save('../../Experiment/SH_MVDR/input_data/pos_sources.mat', 'pos_sources');

% Saving frequency array (to create a tensor in python)
save('../../Experiment/SH_MVDR/input_data/freq.mat', 'freq_array');

% Saving the recorded signal
save('../../Experiment/SH_MVDR/input_data/y.mat', 'y_noise');


%% Recorded signals

% Time vector
dura1 = length(s1) / fs;
time1 = linspace(0, dura1, length(s1));

% Each microphone
mic1 = y(:, 1);
mic2 = y(:, 2);
mic3 = y(:, 19);

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
% SPECTROGRAM FOR SOURCES
winlen = uint32(128); % winlen(samples) = fs * winlen(s)
hop = 0.5;  % Overlap. Default is 50%, or 0.5
nfft = 128; % Default is same length as winlen
% Fourier transform for each window
stftObj = STFTClass(fs, winlen, hop, nfft);

s1_STFT = stftObj.stft(s1, T);
s1_psd = 10*log10(abs(s1_STFT));
figure;
imagesc(T_vector, freq_array, s1_psd);
axis xy;
colorbar_handle = colorbar; 
xlabel('Timeframes');
ylabel('Frequency (Hz)');
title('Source 1 PSD');
ylabel(colorbar_handle, 'PSD(dB/Hz)');
colormap('hot');
caxis([-70, 0]);