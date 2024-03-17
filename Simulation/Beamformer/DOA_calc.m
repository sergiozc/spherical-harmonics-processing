function [doa] = DOA_calc(sensor_pos,source_pos)
% Calcula la DOA de la señal procedente de la fuente dado un sensor de 
% referencia.
% Se calcula el ángulo (radianes) entre dos puntos (vectores) mediante
% la fórmula del coseno.

num = dot(sensor_pos, source_pos);
denom = norm(sensor_pos) * norm(source_pos);

doa = acos(num / denom);

end

