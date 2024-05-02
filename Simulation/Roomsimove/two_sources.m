clear all;
close all;
clc;

addpath(genpath('stft_library'))

%% Sources definition
% FIRST SOURCE
[s1,fs] = audioread('sources/every_man_De_Niro.wav');
s1 = s1(1:92708);  % Same length as source 2
% Position (in meters)
x1 = 1;
y1 = 3.5;
z1 = 1.8; %eg: human height

% SECOND SOURCE
[s2,fs] = audioread('sources/eastwood_lawyers.wav');
% Position (in meters)
x2 = 4;
y2 = 1.5;
z2 = 1.7; %eg: another human height

%% Sources visualization
% TIME VISUALIZATION

% Time vector
dura1 = length(s1) / fs;
time1 = linspace(0, dura1, length(s1));
dura2 = length(s2) / fs;
time2 = linspace(0, dura2, length(s2));

figure(1);
plot(time1, s1);
hold on;
plot(time2, s2);
hold off;
xlabel('Time (s)');
ylabel('Amplitude');
title('Sources in time domain');
legend('Source 1', 'Source 2');

% FREQUENCY DOMAIN
s1_f = fft(s1);
s2_f = fft(s2);
freq = linspace(0, fs, length(s1_f));

figure(2);
plot(freq(1:length(s1_f)/2), abs(s1_f(1:length(s1_f)/2)));
hold on;
plot(freq(1:length(s1_f)/2), abs(s2_f(1:length(s1_f)/2)));
hold off;
grid on;
title('Sources in frequency domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
legend('Source 1', 'Source 2');

% SPATIAL VISUALIZATION
scatter3(x1, y1, z1);
hold on
scatter3(x2, y2, z2);
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Sources positions');

%% Simulation

H1 = roomsimove_single('room_sensor_config.txt',[x1; y1; z1]);
H2 = roomsimove_single('room_sensor_config.txt',[x2; y2; z2]);
y = fftfilt(H1,s1) + fftfilt(H2,s2);

%% Sound pressure
winlen = uint32(256); % it means 256 samples. 
% winlen = 256; % It means 256 ms
hop = 0.25;  % 75% overlap. Default is 50%, or 0.5
nfft = 512; % Default is same length as winlen

% Transformada de fourier para cada ventan
stftObj = STFTClass(fs, winlen, hop, nfft);

% Perform the STFT on y
T = 500; % Number of time frames

% Sound pressure for each microphone
P1 = stftObj.stft(y(:, 1), T);
P2 = stftObj.stft(y(:, 2), T);
P3 = stftObj.stft(y(:, 3), T);
freq_array = stftObj.freqArray;
%% Saving data
%save('../Beamformer/input/y_recorded.mat', 'y');
% Posiciones x, y, z de los sensores y de las fuentes 
% COMPROBAR QUE SON LOS MISMOS DATOS QUE 'room_sensor_config.txt'
pos = [2, 0.5, 1.4; 2, 1, 1.4; 2, 1.5, 1.4; x1, y1, z1; x2, y2, z2];
%save('../Beamformer/input/positions.mat', 'pos');

% Saving frequency array (to create a tensor in python)
save('../PSD-algorithm/data/freq.mat', 'freq_array');


%% Recorded signals

% Each microphone
mic1 = y(:, 1);
mic2 = y(:, 2);
mic3 = y(:, 3);

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
title('Recorded signals');
legend('Mic 1', 'Mic 2', 'Mic 3');

