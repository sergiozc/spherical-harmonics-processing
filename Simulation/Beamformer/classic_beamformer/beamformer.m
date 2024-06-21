%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% BEAMFORMER IMPLEMENTATION %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sergio Zapata Caparrós
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MIT License
% Copyright (c) [2023]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc;
clear all;
close all;


%% Load data
nsensors = 3; % number o sensors
ncent = ceil(nsensors / 2);
y = load('input/y_recorded.mat').y;
% Normalization
maxmax=max(max(abs(y)));
y=y/maxmax;

% Source
[source1, fs] = audioread('input/source1.wav');
source1 = source1(1:92708);
% Normalization
maxmax = max(max(abs(source1)));
source1 = source1/maxmax;

% DOA calculation
positions = load('input/positions.mat').pos;
pos_sensor_cent = positions(ncent, :);
pos_sensor_ini = positions(1, :);
% Two sources
pos_source1 = positions(4, :);
pos_source2 = positions(5, :);

DOA = DOA_calc(pos_sensor_ini, pos_source1);

%% Definición de variables

Fs = 11025;         % sample rate
d = 0.5;            % distance between the elemts of the array
Vprop = 340;        % sound velocity
Ltrama = 256;       % Frames of 256 samples
Lfft = 512;         % FFT length
N = nsensors;       % Number of sensors
phi = DOA;          % Direction of arrival (DOA)
L_signal = length(y(:,ncent));   % Signal length
win = hanning(Ltrama+1,'periodic'); % Hanning window 
freq = linspace(0,256,257)*(Fs/Lfft); % Frequency array
n=0:1:N-1; % Index of the elements of the array
c = 340; % propagation velocity

% Visualization
dura_cent = length(y(:,ncent)) / Fs;
time1 = linspace(0, dura_cent, L_signal);
figure(1)
plot(time1, y(:,ncent))
title('Central sensor')
ylabel('Amplitude');
xlabel('Time (s)');

%% Wave type definition
% Flat or spherical can be chosen
[d_n, tn] = wave_type(c, d, N, phi, 'spherical');


%% Beamforming implementation

% Weights calculation (delay and sum)
w = DAS_weights(d_n, tn, freq); fprintf('Beamformer: DAS \n');
 
% Divisible signal into Ltrama
[m,~] = size(y);
resto = mod(m,Ltrama);
y = y(1:m-resto,:);
[m,~] = size(y); 

Ntramas = 2*(m/Ltrama)-1;

%% Frames processing (analysis-synthesis)

xc_out = zeros(L_signal,N); % Final matrix
XOUT = zeros(Lfft/2+1, 1); % Signal after applying weights
iter = 1;
for ntram = 1:Ntramas  % each frame

    for c = 1:N        % each sensor
        
        xn = y(iter:iter + Ltrama ,c); % a piece of the signal
        Xn = fft(win.*xn, Lfft);        % We perform the Fourier transform of the window (512 samples)
        Xn = Xn(1:Lfft/2+1);          % We take the frequency components from 0 to Fs/2    
        Xn = Xn .* conj(w(:,c));        % Multiplying by the weights
        

        % Forcing symmetry
        simet = conj(Xn);
        XOUT = cat(1, Xn, simet(end:-1:2));
        xout = real(ifft(XOUT));
        
        % Frame concatenation''overlap add''
        xc_out(iter:iter + Lfft, c) = xc_out(iter:iter + Lfft, c) + xout;

    end
    
    iter = iter + (Ltrama/2-1);
end

% Joining all channels
xc_out_sum = sum(xc_out, 2);
% Eliminating residual tail of the last frame
xc_out_sum = xc_out_sum(1:end-Lfft/2);
% Normalization and listening
xout_norm = xc_out_sum/max(abs(xc_out_sum));
soundsc(real(xout_norm),Fs);

% Saving the signal
fout=strcat('./results/DAS_result','.wav');
audiowrite(fout, xout_norm, Fs)


figure(2)
plot(y(:,ncent));
hold on
plot(real(xout_norm));
hold off
legend('Central sensor signal','Beamformer output signal')
title('Time representation after beamforming')
%Se puede comprobar como el ruido se ha minimizado


%% Correlation metrix

% BEFORE BEAMFORMING
% Correlation matrix
corr_matrix2 = corrcoef(source1(1:length(y(:,1))), y(:,1));
% Correlation between both signals
corr_coef2 = corr_matrix2(1, 2);

% AFTER BEAMFORMING
% Correlation matrix
corr_matrix1 = corrcoef(source1(1:length(xout_norm)), xout_norm);
% Correlation between both signals
corr_coef1 = corr_matrix1(1, 2);

fprintf('Correlation coefficient before BF = %f\n', corr_coef2);
fprintf('Correlation coefficient after BF = %f\n', corr_coef1);

%% SNR metrix (not useful...)

% Source power
power_source = var(source1);
% Input power (signal mixed)
power_input = var(y(:, 1));
% Output (after beamforming) power
power_output = var(xout_norm);
% The diference between the input power and the source power is the noise
% power
power_noise = max(power_input - power_source, 0);

% SNR before beamforming
SNR_orig = SNR_calc(power_input, power_noise);
fprintf('SNR(before)  = %f dB\n', SNR_orig);

% SNR after beamforming
SNR_BF = SNR_calc(power_output, power_noise);
fprintf('SNR(after)  = %f dB\n', SNR_BF);