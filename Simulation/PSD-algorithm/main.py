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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# %%
# PARAMETER DEFINITION

c = 343 # Propagation speed
r = 0.042 # Assuming same radius
room_size = np.array([5, 4, 2.6]) # Room dimensions

# Microphones' posictions
pos_mic = loadmat('data/pos_mic.mat')['pos_mic'] # Microphones positions (x,y,z)
Q = pos_mic.shape[0]  # Number of microphones (selecting rows' length) 
el = np.zeros(Q) # Mic elevation
az = np.zeros(Q) # Mic azimut
r_pos_mic = np.zeros(Q) # Distance vector for microphones
for i, (x, y, z) in enumerate(pos_mic): # Switch to el and az (from cartesian)''
    el[i], az[i], r_pos_mic[i] = utils.cart2sph(x, y, z)

# Sources' positions
pos_sources = loadmat('data/pos_sources.mat')['pos_sources'] # Sources positions (x,y,z)
L = pos_sources.shape[0]  # Number of sources (selecting rows' length)
el_s = np.zeros(L) # Sources elevation
az_s = np.zeros(L) # Sources azimut
r_pos_s = np.zeros(L) # Distance vector (r) for sources
for j, (x_s, y_s, z_s) in enumerate(pos_sources):# Switch to el and az (from cartesian)
    el_s[j], az_s[j], r_pos_s[j] = utils.cart2sph(x_s, y_s, z_s)

# Frequency values
freq_array = loadmat('data/freq.mat')['freq_array'] # Frequency array
freq_array[0] = 1 # To avoid dividing by zero
k_array = SHutils.getK(freq_array, c) # Wavenumber array
Nfreq = len(k_array) # Number of frequencies

# Sound pressure and timeframes
P = loadmat('data/sound_pressure.mat')['P'] # Sound pressure
timeFrames = P.shape[2] # Number of time frames


order = 4 # Spherical harmonic order (it defines the complexity of the spherical function).
# Defined in the "experimental setup" of the paper. Highest order to calculate.
Nmin = 2 # Minimun order

# V <= int(np.floor(np.sqrt((order + 1)**2 - L - 1) - 1)) # Order of the power of a reverberation sound field.
V = 1
# See Eq (59) from the paper. # The higher the order the greater the reverberation complexity.

# Order and Mode array
nm_arr = SHutils.ACNOrderModeArray(order)
# Index to cut from Nmin. E.g to get alpha from Nmin.
cut_Nmin = Nmin**2

# %% Microphones and sources spatial visualization (check)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos_mic[:, 0], pos_mic[:, 1], pos_mic[:, 2], c='b', marker='o', label='Microphones')
ax.scatter(pos_sources[0, 0], pos_sources[0, 1], pos_sources[0, 2], c='r', marker='o', label='Source 1')
ax.scatter(pos_sources[1, 0], pos_sources[1, 1], pos_sources[1, 2], c='r', marker='o', label='Source 2')
ax.scatter(pos_sources[2, 0], pos_sources[2, 1], pos_sources[2, 2], c='r', marker='o', label='Source 3')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_xlim([0, room_size[0]])
ax.set_ylim([0, room_size[1]])
ax.set_zlim([0, room_size[2]])
ax.set_title('Microphones and sources positions')
plt.show()


#%%
# SH calculations (just to test if necessary)

# Real spherical harmonics for n = 3 and m = 2. For all elevations and azimuths.
#y2 = SHutils.realSHPerMode(3, 2, el, az)  # [Q x 1] output vector
# Complex spherical harmonics for n = 3, m = 2 for all elevations and azimuths.
#y4 = SHutils.complexSHPerMode(3, 2, el_s, az_s) # [Q x 1] output vector

# Complex spherical harmonics upto order = order. For all elevations and azimuths.
#y3 = SHutils.complexSH(order, el, az)

# Get all SH orders in an array according to ACN (Ambisonics Channel Numbering)
#n_arr = SHutils.ACNOrderArray(order)


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
# Frequency loop. Alpha is calculated for nm (see nm_array)
for k_index, k in enumerate(k_array):
    alpha[k_index, :, :] = SHutils.alphaOmni(P[k_index, :, :], order, k, r, el, az, isRigid, mode, compensateBesselZero, applyRegularization, weights)

# Calculated alpha from Nmin
alpha_cut = alpha[:, cut_Nmin:, :]
#%% 
# UPSILON, PSI, OMEGA calculation (3rd step of the algorithm)

#freq = 882.861 # Frequency (unit test)
#k = SHutils.getK(freq, c)

#ups = PSDestimation.upsilon_term(0, 0, 0, 0, el_s, az_s)
#psi = PSDestimation.psi_term(0, 0, 0, 0, 2, 1)
#omega = PSDestimation.omega_term(0, 0, 0, 0, k, r)

#%%
# PSDs MATRIX CALCULATION

N = order # order
# Translation matrix (T) (tensor 3D)
T_matrix = PSDestimation.translation_matrix(el_s, az_s, k_array, r, N, Nmin, V, L)

# Lambda matrix (tensor 3D)
beta = 0.3 # Smoothing factor
lambda_matrix_t  = PSDestimation.lambda_matrix(k_array, N, Nmin, beta, alpha_cut, timeFrames)

# %%
# PSD matrix (per time frames)
theta_psd = np.zeros((Nfreq, (L + (V + 1)**2 + 1), timeFrames), dtype=complex)
for i in range (timeFrames):
    theta_psd[:, :, i] = PSDestimation.psd_matrix(T_matrix, lambda_matrix_t[:, :, i])

# the first L elements of the theta_psd matrix (second dimension) represent the estimated source PSDs at the origin
# the last element represents the estimated noise PSD at the origin. (second dimension)

#%%
# PSD representation

# Heatmap for noise
# noise_psd = np.abs(theta_psd[:, -1, :])
# plt.figure()
# plt.imshow(10*np.log10(noise_psd), aspect='auto', cmap='hot', origin='lower',extent=[0, timeFrames-1, 0, int(freq_array[-1])], vmin=-70, vmax=0)
# plt.colorbar(label='PSD(dB/Hz)')
# plt.xlabel('Time frames')
# plt.ylabel('Frequency (Hz)')
# plt.title('Estimated PSD. Noise.')
# plt.show()

# Heatmap for first source
source1_psd = np.abs(theta_psd[:, 0, :])
plt.figure()
plt.imshow(10*np.log10(source1_psd), aspect='auto', cmap='hot', origin='lower',extent=[0, timeFrames-1, 0, int(freq_array[-1])], vmin=-70, vmax=0)
plt.colorbar(label='PSD(dB/Hz)')
plt.xlabel('Time frames')
plt.ylabel('Frequency (Hz)')
plt.title('Estimated PSD. Source 1.')
plt.show()
# Heatmap for second source
source2_psd = np.abs(theta_psd[:, 1, :])
plt.figure()
plt.imshow(10*np.log10(source2_psd), aspect='auto', cmap='hot', origin='lower',extent=[0, timeFrames-1, 0, int(freq_array[-1])], vmin=-70, vmax=0)
plt.colorbar(label='PSD(dB/Hz)')
plt.xlabel('Time frames')
plt.ylabel('Frequency (Hz)')
plt.title('Estimated PSD. Source 2.')
plt.show()
# Heatmap for second source
source3_psd = np.abs(theta_psd[:, 2, :])
plt.figure()
plt.imshow(10*np.log10(source3_psd), aspect='auto', cmap='hot', origin='lower',extent=[0, timeFrames-1, 0, int(freq_array[-1])], vmin=-70, vmax=0)
plt.colorbar(label='PSD(dB/Hz)')
plt.xlabel('Time frames')
plt.ylabel('Frequency (Hz)')
plt.title('Estimated PSD. Source 3')
plt.show()


# reverberation = np.abs(theta_psd[:, 3, :])
# plt.figure()
# plt.imshow(10*np.log10(reverberation), aspect='auto', cmap='hot', origin='lower',extent=[0, timeFrames-1, 0, int(freq_array[-1])], vmin=-70, vmax=0)
# plt.colorbar(label='PSD(dB/Hz)')
# plt.xlabel('Time frames')
# plt.ylabel('Frequency (Hz)')
# plt.title('Estimated PSD. Reverberation 1.')
# plt.show()
