%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%script to visualize the spectrum%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
clc;
close all;

% Source
[source1, fs] = audioread('input/source1.wav');
source1 = source1(1:92708);
% Normalization
maxmax = max(max(abs(source1)));
source1 = source1/maxmax;


[pxx,f] = pwelch(source1,500,300,500,fs);

figure(1)
plot(f,10*log10(pxx))
xlabel('Frequency (Hz)')
ylabel('PSD (dB/Hz)')
title('Frequency spectrum (source)')
grid on