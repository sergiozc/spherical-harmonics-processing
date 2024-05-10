# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 18:22:30 2024

@author: sergiozc
@references: Fahim, Abdullah, et al. “PSD Estimation and Source Separation in a Noisy Reverberant Environment Using a Spherical Microphone Array.” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 26, no. 9, Institute of Electrical and Electronics Engineers (IEEE), Sept. 2018, pp. 1594–607, doi:10.1109/taslp.2018.2835723.
"""

from sphericalHarmonic import SHutils
from PSD_estimation import PSDestimation
from utils import utils
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
# %%
# PARAMETER DEFINITION

r = 0.042 # Assuming same radius
c = 343 # Propagation speed

pos_mic = loadmat('data/pos_mic.mat')['pos_mic'] # Microphones positions (x,y,z)
Q = pos_mic.shape[0]  # Number of microphones (selecting rows' length) 
el = np.zeros(Q) # Mic elevation
az = np.zeros(Q) # Mic azimut
for i, (x, y, z) in enumerate(pos_mic): # Switch to el and az (from cartesian)
    el[i], az[i] = utils.cartesian2ElAz(x, y, z)

pos_sources = loadmat('data/pos_sources.mat')['pos_sources'] # Sources positions (x,y,z)
L = pos_sources.shape[0]  # Number of sources (selecting rows' length)
# Sources' configuration
el_s = np.linspace(0, np.pi, L)  # Sources elevation [0, pi] (rad). 
az_s = np.linspace(0, 2 * np.pi, L)  # # Microphones azimuth [0, pi] (rad). 

el_s = np.zeros(L) # Sources elevation
az_s = np.zeros(L) # Sources azimut
for i, (x, y, z) in enumerate(pos_sources):# Switch to el and az (from cartesian)
    el_s[i], az_s[i] = utils.cartesian2ElAz(x, y, z)


freq_array = loadmat('data/freq.mat')['freq_array'] # Frequency array
freq_array[0] = 0.1 # To avoid dividing by zero
k_array = SHutils.getK(freq_array, c) # Wavenumber array
Nfreq = len(k_array) # Number of frequencies


P = loadmat('data/sound_pressure.mat')['P'] # Sound pressure
P = np.transpose(P, (0, 2, 1)) # Adjust to a (freq x Q x time) tensor
timeFrames = P.shape[2] # Number of time frames


order = 4 # Spherical harmonic order (it defines the complexity of the spherical function).
# Defined in the "experimental setup" of the paper. Highest order to calculate.
Nmin = 2 # Minimun sound order
V = int(np.floor(np.sqrt((order + 1)**2 - L - 1) - 1)) # Order of the power of a reverberation sound field.
# See Eq (59) from the paper. # The higher the order the greater the reverberation complexity.

#%%
# SH calculations (just to test if necessary)

# Real spherical harmonics for n = 3 and m = 2. For all elevations and azimuths.
y2 = SHutils.realSHPerMode(3, 2, el, az)  # [Q x 1] output vector
# Complex spherical harmonics for n = 3, m = 2 for all elevations and azimuths.
y4 = SHutils.complexSHPerMode(3, 2, el, az) # [Q x 1] output vector

# Complex spherical harmonics upto order = order. For all elevations and azimuths.
y3 = SHutils.complexSH(order, el, az)

# Get all SH orders in an array according to ACN (Ambisonics Channel Numbering)
n_arr = SHutils.ACNOrderArray(order)


# %%
# ALPHA (sound field coefficients) for all modes and time frames (2nd step of the algorithm)

# Mode: no-inv is the same as Ec(12) within “PSD Estimation and Source Separation in a Noisy Reverberant Environment Using a Spherical Microphone Array.” 
#mode = 'inv'
mode = 'no-inv'
# Weights to the corresponding microphones
#(used if mode ~= inv)
#weight = np.random.rand(32) + 0.01
#weight = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.5, 0.3, 0.3, 0.3, 0.5, 0.3, 0.3, 0.3, 0.5, 0.6, 0.7, 0.7, 0.7, 0.6, 0.7, 0.7, 0.7])
weights = np.ones(Q)
# Rigid array or not
isRigid = True
# To compensate bessel 0 issue or not
#(used if mode != inv)
compensateBesselZero = True

# To apply regularization or not
# (used if mode == inv)
applyRegularization = False

# Alpha as a tensor (depends on frequency)
alpha = np.zeros((Nfreq, (order+1)**2, timeFrames), dtype=complex)
# Frequency loop
for k_index, k in enumerate(k_array):
    alpha[k_index, :, :] = SHutils.alphaOmni(P[k_index, :, :], order, k, r, el, az, isRigid, mode, compensateBesselZero, applyRegularization, weights)
    
#%% 
# UPSILON, PSI, OMEGA calculation (3rd step of the algorithm)

#freq = 882.861 # Frequency (unit test)
#k = SHutils.getK(freq, c)

#ups = PSDestimation.upsilon_term(0, 0, 0, 0, el_s, az_s)
#psi = PSDestimation.psi_term(0, 0, 0, 0, 2, 1)
#omega = PSDestimation.omega_term(0, 0, 0, 0, k, r)

#%%
# PSDs MATRIX CALCULATION

N = 4 # order
# Translation matrix (T) (tensor 3D)
T_matrix = PSDestimation.translation_matrix(el_s, az_s, k_array, r, N, V, L)

# Lambda matrix (tensor 3D)
beta = 0.8 # Smoothing factor
lambda_matrix_t  = PSDestimation.lambda_matrix(k_array, N, beta, alpha, timeFrames)

# %%
# PSD matrix (per time frames)
theta_psd = np.zeros((Nfreq, (L + (V + 1)**2 + 1), timeFrames), dtype=complex)
for i in range (timeFrames):
    theta_psd[:, :, i] = PSDestimation.psd_matrix(T_matrix, lambda_matrix_t[:, :, i])

# the first L elements of the theta_psd matrix (second dimension) represent the estimated source PSDs at the origin
# the last element represents the estimated noise PSD at the origin. (second dimension)

#%% PSD representation
noise_psd = np.abs(theta_psd[:, -1, :]).T
# Making the heatmap
plt.figure()
plt.imshow(noise_psd, aspect='auto', cmap='viridis', origin='lower',extent=[0, timeFrames-1, 0, 5500])
plt.colorbar(label='PSD(dB/Hz)')
plt.xlabel('Time frames')
plt.ylabel('Frequency (Hz)')
plt.title('Noise PSD heatmap')
plt.show()


