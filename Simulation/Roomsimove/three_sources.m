
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT WHICH PERFORMS THE SIMULATION FOR 3 SOURCES AND CALCULATES THE
% SOUND PRESSURE.
% Author: sergiozc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

addpath(genpath('stft_library'))
addpath(genpath('SH_library'))

%% Sources definition (fs in "room_sensor_config.txt too)

fs = 16000; % 16kHz

% FIRST SOURCE (voice)
[s1,fs_voice] = audioread('sources/scarface_alpacino.wav'); % VOICE
s1 = resample(s1, fs, fs_voice); % Resample to conver fs to 16kHz % VOICE
% Position (in meters)
x1 = 4.9;
y1 = 0.1;
z1 = 1.65; %eg: human height


% SECOND SOURCE 
%[s2,fs_voice] = audioread('sources/every_man_De_Niro.wav'); % VOICE
%s2 = resample(s2, fs, fs_voice); % Resample to converfs to 16kHz % VOICE
freq2 = 800; duration = 9; % Hz, samples/s, s % SIN
s2 = sin_source_generation(freq2, duration, fs); % SIN
s2 = s2(1:length(s1));
% Position (in meters)
x2 = 0.1;
y2 = 3.9;
z2 = 2.5; %Located in a corner

% THIRD SOURCE (sin)
%[s3,fs_voice] = audioread('sources/eastwood_lawyers.wav'); % VOICE
%s3 = resample(s3, fs, fs_voice); % Resample to conver fs to 16kHz % VOICE
freq3 = 2500; duration = 9; % Hz, samples/s, s % SIN
s3 = sin_source_generation(freq3, duration, fs); % SIN
s3 = s3(1:length(s1));
% Position (in meters)
x3 = 4.9;
y3 = 3.9;
z3 = 2.5; % Located in the corner

%% Sources visualization
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

% Extract mic position from eigenmike HOM
hom = SHTools.getEigenmike();
cart_eigen = hom.cart; % Cartesians
% Center of the sphere
center = [1, 1, 1];
% Shift the geometry 
pos_mic = cart_eigen + center;

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

% Received signal at qth microphone
y = fftfilt(H1,s1) + fftfilt(H2,s2) + fftfilt(H3,s3);

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

for n = 1:Nmic
    % STFT in each microphone signal
    P(:, n, :) = stftObj.stft(y(:, n), T);
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
save('../PSD-algorithm/data/pos_mic.mat', 'pos_mic');
%save('../../Experiment/SH_MVDR/input_data/pos_mic.mat', 'pos_mic');

% Saving sources positions (cartesian)
pos_sources = [x1, y1, z1; x2, y2, z2; x3, y3, z3];
save('../PSD-algorithm/data/pos_sources.mat', 'pos_sources');
%save('../../Experiment/SH_MVDR/input_data/pos_sources.mat', 'pos_sources');

% Saving frequency array (to create a tensor in python)
%save('../../Experiment/SH_MVDR/input_data/freq.mat', 'freq_array');
save('../PSD-algorithm/data/freq.mat', 'freq_array');

% Saving sound pressure tensor
%save('../../Experiment/SH_MVDR/input_data/sound_pressure.mat', 'P');
save('../PSD-algorithm/data/sound_pressure.mat', 'P');

% Saving the recorded signal
%save('../../Experiment/SH_MVDR/input_data/y.mat', 'y');


%% Recorded signals

% Time vector
dura1 = length(s1) / fs;
time1 = linspace(0, dura1, length(s1));
dura2 = length(s2) / fs;
time2 = linspace(0, dura2, length(s2));
dura3 = length(s3) / fs;
time3 = linspace(0, dura3, length(s3));

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

s2_STFT = stftObj.stft(s2, T);
s2_psd = 10*log10(abs(s2_STFT));
figure;
imagesc(T_vector, freq_array, s2_psd);
axis xy;
colorbar_handle = colorbar; 
xlabel('Timeframes');
ylabel('Frequency (Hz)');
title('Source 2 PSD');
ylabel(colorbar_handle, 'PSD(dB/Hz)');
colormap('hot');
caxis([-70, 0]);

s3_STFT = stftObj.stft(s3, T);
s3_psd = 10*log10(abs(s3_STFT));
figure;
imagesc(T_vector, freq_array, s3_psd);
axis xy;
colorbar_handle = colorbar;  
xlabel('Timeframes');
ylabel('Frequency');
title('Source 3 PSD');
ylabel(colorbar_handle, 'PSD(dB/Hz)');
colormap('hot');
caxis([-70, 0]);
