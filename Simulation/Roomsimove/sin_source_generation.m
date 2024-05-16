function [source] = sin_source_generation(freq, duration, fs)
    % Definir el tiempo
    time = linspace(0, duration, fs * duration);
    % Generating the signal
    source = sin(2*pi*freq*time);
    % Normalization
    source = source / max(abs(source));
end

