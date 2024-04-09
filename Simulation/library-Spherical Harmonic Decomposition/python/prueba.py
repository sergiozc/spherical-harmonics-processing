# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 18:22:30 2024

@author: sergiozc
@references: Fahim, Abdullah, et al. “PSD Estimation and Source Separation in a Noisy Reverberant Environment Using a Spherical Microphone Array.” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 26, no. 9, Institute of Electrical and Electronics Engineers (IEEE), Sept. 2018, pp. 1594–607, doi:10.1109/taslp.2018.2835723.
"""

from sphericalHarmonic import SHutils
import numpy as np

# %% PARAMETER DEFINITION (valores de prueba)
Q = 32  # Number of microphones
freq = 1000 # Frequency
c = 343 # Propagation speed
order = 4 # Spherical harmonic order (it defines the complexity of the spherical function)
el = np.linspace(0, np.pi, Q) # Microphones elevation [0, pi] (rad). 
az = np.linspace(0, 2*np.pi, Q) # Microphones azimuth [0, 2*pi] (rad)
r = np.full((32, 1), 0.042) # Radial distances (to the origin of the array)

K = 5 # Number of sources
timeFrames = 500 

# Sources' configuration
el_s = np.linspace(0, np.pi, K)  # Sources elevation [0, pi] (rad). 
az_s = np.linspace(0, 2 * np.pi, K)  # # Microphones azimuth [0, pi] (rad). 
r_s = np.linspace(0, 2, K)  # Radius [0,2]

# Sources' amplitudes
amp_s = (np.linspace(0, 10, K)[:, np.newaxis] * np.ones((K, timeFrames))).T  # Amplitudes [0 ,10]

# Spherical harmonics order
order = 4 # It defines the complexity of the spherical function

# Sound pressure
# Crea la matriz compleja P con los valores especificados
P = np.ones((32, 500)) + 1j * np.zeros((32, 500))


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
# ALPHA (sound field coefficients) for all modes and time frames

# Wavenumber calculation
k = SHutils.getK(freq, c)

# Mode: no-inv is the same as Ec(12) within “PSD Estimation and Source Separation in a Noisy Reverberant Environment Using a Spherical Microphone Array.” 
#mode = 'inv'
mode = 'no-inv'

# Weights to the corresponding microphones
#(used if mode ~= inv)
#weight = np.random.rand(32) + 0.01
#weight = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.5, 0.3, 0.3, 0.3, 0.5, 0.3, 0.3, 0.3, 0.5, 0.6, 0.7, 0.7, 0.7, 0.6, 0.7, 0.7, 0.7])
weights = np.ones(Q)

# Highest SH order to calculate.
# # It defines the complexity of the spherical function
order = 3

# Rigid array or not
isRigid = True

# To compensate bessel 0 issue or not
#(used if mode != inv)
compensateBesselZero = True

# To apply regularization or not
# (used if mode == inv)
applyRegularization = False

alpha = SHutils.alphaOmni(P, order, k, r, el, az, isRigid, mode, compensateBesselZero, False, weights)
