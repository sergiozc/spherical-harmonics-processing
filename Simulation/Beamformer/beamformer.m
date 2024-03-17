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
fs = 11025;
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

Fs = 11025;         % Frecuencia de muestreo
d = 0.5;            % Separación entre elementos del array (m)
Vprop = 340;        % Velocidad del sonido
Ltrama = 256;       % Tramas de 256 muestras
Lfft = 512;         % Longitud de la FFT
N = nsensors;       % Número de sensores
phi = DOA;         % Ángulo de llegada del target (DOA)
L_signal = length(y(:,ncent));   %Longitud total de la señal
win = hanning(Ltrama+1,'periodic'); % Ventana de hanning  
freq = linspace(0,256,257)*(Fs/Lfft); % Vector de frecuencias
n=0:1:N-1; % Índice de los elementos del array
c = 340; % Velocidad de propagación

% Visualization
dura_cent = length(y(:,ncent)) / Fs;
time1 = linspace(0, dura_cent, L_signal);
figure(1)
plot(time1, y(:,ncent))
title('Central sensor')
ylabel('Amplitude');
xlabel('Time (s)');

%% Tipo de onda
% Se puede elegir entre onda plana o esférica
[d_n, tn] = wave_type(c, d, N, phi, 'spherical');


%% Cómputo del Beamformer

% CÁLCULO DE PESOS
% Pesos DAS
w = DAS_weights(d_n, tn, freq); fprintf('Beamformer: DAS \n');

% SEÑAL DIVISIBLE ENTRE Ltrama 
[m,~] = size(y);
resto = mod(m,Ltrama);
y = y(1:m-resto,:);
% Se obtiene el número de muestras que tendrá la señal sobre la que se
% aplicará el beamforming
[m,~] = size(y); 

Ntramas = 2*(m/Ltrama)-1;

%% PROCESADO POR TRAMAS (análisis-síntesis)

xc_out = zeros(L_signal,N); % Matriz del resultado final
XOUT = zeros(Lfft/2+1, 1); % Señal depúes de aplicar los pesos
iter = 1;
for ntram = 1:Ntramas  % Se computa cada trama

    for c = 1:N        % Se computa cada sensor
        
        xn = y(iter:iter + Ltrama ,c); %Tomamos la porción de señal del canal correspondiente
        Xn = fft(win.*xn, Lfft);        %Realizamos la transformada de Fourier de la ventana (512 muestras)
        Xn = Xn(1:Lfft/2+1);          %Tomamos las componentes de frecuencia de 0 a Fs/2 (Fs/2 = 8 kHz)s     
        Xn = Xn .* conj(w(:,c));        %Multiplicamos por los pesos correspondientes
        

        %Realizamos la simetrización para practicar la transformada inversa
        simet = conj(Xn);
        XOUT = cat(1, Xn, simet(end:-1:2));
        xout = real(ifft(XOUT));
        
        %Concatenación de tramas mediante ''overlap add''
        xc_out(iter:iter + Lfft, c) = xc_out(iter:iter + Lfft, c) + xout;

    end
    
    iter = iter + (Ltrama/2-1);
end

%Unimos todos los canales
xc_out_sum = sum(xc_out, 2);
% Eliminamos la cola residual de la ultima trama
xc_out_sum = xc_out_sum(1:end-Lfft/2);
% Normalizamos la señal y la escuchamos
xout_norm = xc_out_sum/max(abs(xc_out_sum));
soundsc(real(xout_norm),Fs);

% Guardamos señal resultante normalizada
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


%% SALE LA POTENCIA DEL RUIDO NEGATIVA!!! REVISAR
%% Cálculo SNR

% Calcular la potencia de la señal deseada
power_source = var(source1);
% Calcular la potencia de la señal de entrada
power_input = var(y(:, 1));
% Calcular la potencia de la señal de salida del beamformer
power_output = var(xout_norm);
% Calcular la diferencia entre la potencia de la señal deseada y la de
% entarda
power_noise = max(power_input - power_output, 0);

% SNR ANTES DEL BEAMFORMING
SNR_orig = SNR_calc(power_input, power_noise);
fprintf('SNR(before)  = %f dB\n', SNR_orig);

% SNR DESPUÉS DEL BEAMFORMING
SNR_BF = SNR_calc(power_output, power_noise);
fprintf('SNR(after)  = %f dB\n', SNR_BF);
