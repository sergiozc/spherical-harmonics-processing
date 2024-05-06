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
# %%
# PARAMETER DEFINITION (valores de prueba)

Q = 32  # Number of microphones
r = 0.042 # Assuming same radius
freq_array = loadmat('data/freq.mat')['freq_array'] 
freq_array[0] = 0.1 # To avoid dividing by zero
c = 343 # Propagation speed
k_array = SHutils.getK(freq_array, c) # Wavenumber array
Nfreq = len(k_array) # Number of frequencies

freq = 882.861 # Frequency (unit test)
k = SHutils.getK(freq, c)

order = 4 # Spherical harmonic order (it defines the complexity of the spherical function).
# Defined in the "experimental setup" of the paper. Highest order to calculate.

Nmin = 2 # Minimun sound order

el = np.linspace(0, np.pi, Q) # Microphones elevation [0, pi] (rad). 
az = np.linspace(0, 2*np.pi, Q) # Microphones azimuth [0, 2*pi] (rad)
r = np.full((Q, 1), 0.042) # Radial distances (to the origin of the array).
# If radius distances are the same. Must be a number.

# Using cartesian coordenates
# el, az = utils.cartesian2ElAz(1, 4, -1)

L = 5 # Number of sources
timeFrames = 500 # Number of time frames
V = int(np.floor(np.sqrt((order + 1)**2 - L - 1) - 1)) # Order of the power of a reverberation sound field.
# See Eq (59) from the paper. # The higher the order the greater the reverberation complexity.


# Sources' configuration
el_s = np.linspace(0, np.pi, L)  # Sources elevation [0, pi] (rad). 
az_s = np.linspace(0, 2 * np.pi, L)  # # Microphones azimuth [0, pi] (rad). 
r_s = np.linspace(0, 2, L)  # Radius [0,2]

# Sources' amplitudes
amp_s = (np.linspace(0, 10, L)[:, np.newaxis] * np.ones((L, timeFrames))).T  # Amplitudes [0 ,10]

# Sound pressure (este dato lo tenemos que sacar del experimento)
P = np.ones((32, 500)) + 1j * np.zeros((32, 500))

P_vector = np.ones((Q, len(freq_array), timeFrames)) + 1j * np.zeros((Q, len(freq_array), timeFrames))


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
    alpha[k_index, :, :] = SHutils.alphaOmni(P, order, k, r, el, az, isRigid, mode, compensateBesselZero, applyRegularization, weights)
    
#%% 
# UPSILON, PSI, OMEGA calculation (3rd step of the algorithm)

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
