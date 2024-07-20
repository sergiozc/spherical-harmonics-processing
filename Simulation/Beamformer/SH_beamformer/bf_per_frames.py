# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 18:23:21 2024

@author: sergiozc
"""

import numpy as np
from utils import utils
import matplotlib.pyplot as plt
from SH_visualize import SH_visualization
from SH_BF import sphericalBF
from sphericalHarmonic import SHutils
from scipy.io import loadmat
from scipy.fft import fft, ifft
from scipy.signal import hann

plt.close('all')

# %% Parameter definition

SH_order = 3 # Spherical harmonic order (it determines the complexity of the function)
# The higher is the order, the more directive is the beampattern
# Nevertheless, the higuer is the order, the  more spatial aliasing might we have

Nmin = 0 # Lowest order (default 0)
N_harm = (SH_order + 1) ** 2 # Highest number of harmonics (not valid if Nmin != 0)

# Sources' positions
pos_sources = loadmat('input_data/pos_sources.mat')['pos_sources'] # Sources positions (x,y,z)
L = pos_sources.shape[0]  # Number of sources (selecting rows' length)
el_s = np.zeros(L) # Sources elevation
az_s = np.zeros(L) # Sources azimut
r_pos_s = np.zeros(L) # Distance vector (r) for sources
for j, (x_s, y_s, z_s) in enumerate(pos_sources):# Switch to el and az (from cartesian)
    el_s[j], az_s[j], r_pos_s[j] = utils.cart2sph(x_s, y_s, z_s)
    
# FOR TESTING    
#az_s, el_s = sphericalBF.sph_sources(L, 'random') # Sources DOA

Fs = 16000 # Sample rate
Q = 32 # Number of sensors (mics)
Lframe = 256        # Frames of 256 samples
Lfft = 512          # FFT length
y = loadmat('input_data/y.mat')['y'] # Recorded signal
L_signal = len(y[:, 0]) # Signal length
win = hann(Lframe + 1, sym=False)  # Hanning window 
freq = np.linspace(0, 256, 257) * (Fs / Lfft)  # Frequency array

#%% Beamformer

# Spherical harmonics
Y_s = SHutils.realSH(SH_order, utils.el2inc(el_s), az_s) # d_n (steering vector)
# In SH domain, the steering vector of the beamformer is given by the spherical harmonics

# Power of the sources
power_s = np.eye(L) # We assume all sources have the same power

# Crea la matriz diagonal con las potencias
#power_s = np.diag(np.array([1, 3, 5, 1]))

# Power of the noise (diffuse noise)
power_noise = 1

# SH signal covariance matrix (N_harm x N_harm)
cov_matrix = sphericalBF.SH_COV(N_harm, Y_s, power_s, power_noise)

# Metric to evaluate the diference with a perfectly diffuse sound field
dif_mis = sphericalBF.diffuse_mismatch(SH_order, L, cov_matrix)


plt.figure()
plt.imshow(cov_matrix, cmap='viridis')
plt.colorbar()
plt.title('Spherical covariance matrix')
plt.xlabel('SH signal index')
plt.ylabel('SH signal index')
plt.show()

# Weights calculation (DEPENDEN DE LA FRECUENCIA ?!)
w = sphericalBF.SH_MVDR_weights(N_harm, Y_s, cov_matrix)

#%% Frame processing (NO ETSÁ BIEN !!)

# Adjustinig the signal to Lframe
m, _ = y.shape
resto = m % Lframe
y = y[:m-resto, :]
m, _ = y.shape

Nframes = 2 * (m // Lframe) - 1

# Initialization
xc_out = np.zeros((L_signal, Q))  # Final matrix
iter = 0

for ntram in range(Nframes):  # Each frame
    for c in range(Q):  # Each channel (c)
        xn = y[iter:iter + Lframe+1, c]  # Parte of the signal
        Xn = fft(win * xn, Lfft)  # FFT of the window
        Xn = Xn[:Lfft // 2 + 1]  # Getting frequency components from 0 to Fs/2
        Xn = Xn * np.conj(w[:, c])  # Aplying BF weights

        # Forcing symetry
        simet = np.conj(Xn)
        Xout_ = np.concatenate((Xn, simet[-2:0:-1]))
        xout = np.real(ifft(Xout_)) # CHECK THIS FUNCTION OF ifft !!

        # Union of the frames. "Overlap add"
        xc_out[iter:iter + Lfft, c] += xout

    iter += (Lframe // 2 - 1)
    
# Adding all channels
xc_out_sum = np.sum(xc_out, axis=1)
# Deleting the residual tail
xc_out_sum = xc_out_sum[:-(Lfft // 2)]
# Normalization
xout_norm = xc_out_sum / np.max(np.abs(xc_out_sum))

# Recorded signal from one of the microphones
reference_sensor_signal = y[:, 0]
# Comparison of both signals
plt.figure()
plt.plot(reference_sensor_signal, label='Central sensor signal')
plt.plot(np.real(xout_norm), label='Beamformer output signal')
plt.legend()
plt.title('Time representation after beamforming')
plt.show()


# %% BF visualization

# Sources' directions in cartesian coordenates  
src_cart = np.array(utils.sph2cart(1, el_s, az_s)).T # r=1 because is unit vector

# Resolution of the grid
aziRes = 5
elRes = 5

for i in range(L):
    # Visualization with sources' directions
    SH_visualization.plotSphFunctionCoeffs(w[:,i], SH_order, aziRes, elRes)
    plt.title('MVDR BF on source ' + str(i))
    for source in src_cart:
        plt.plot([0, source[0]], [0, source[1]], [0, source[2]], color='black', linewidth=2, linestyle='--')
        
