# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 18:23:50 2024

@author: sergiozc
"""

from scipy.io import loadmat
import numpy as np
from utils import utils
import matplotlib.pyplot as plt
from SH_visualize import SH_visualization
from SH_BF import sphericalBF

# %% Parameter definition

SH_order = 3 # Spherical harmonic order
Nmin = 0 # Lowest order (default 0)
N_harm = (SH_order + 1) ** 2 # Highest number of harmonics (not valid if Nmin != 0)

pos_sources = loadmat('input_data/pos_sources.mat')['pos_sources'] # Sources positions (x,y,z)
L = pos_sources.shape[0]  # Number of sources (selecting rows' length)
el_s = np.zeros(L) # Sources elevation
az_s = np.zeros(L) # Sources azimut
r_pos_s = np.zeros(L) # Distance vector (r) for sources
for j, (x_s, y_s, z_s) in enumerate(pos_sources):# Switch to el and az (from cartesian)
    el_s[j], az_s[j], r_pos_s[j] = utils.cart2sph(x_s, y_s, z_s)

# Sources' Elevation (theta) and Azimut (phi). Direction of the sources (DOAs).
# First source
el_s[0] = 0
az_s[0] = 0

# Second source
el_s[1] = np.pi / 2
az_s[1] = 0

# Third source
el_s[2] = 0
az_s[2] = np.pi / 4

# %% Spherical harmonic visualization
# ¡WARNING!: Many images might be generated

# # # Creating the grid
# theta = np.linspace(0, np.pi, 100)  # Elevation
# phi = np.linspace(0, 2 * np.pi, 100)  # Azimut
# theta, phi = np.meshgrid(theta, phi)

# # Visualization per n and m
# for n_h in range(Nmin, SH_order+1):
#      for m_h in range(-n_h, n_h + 1):
#          SH_visualization.harmonic_plot(SH_order, n_h, m_h, theta, phi)

#%% Beamformer

Y_src = loadmat('input_data/Y_src.mat')['Y_src'] # Spherical harmonics (test data)
#Y_src = SHutils.realSH(SH_order, el_s, az_s)

# In SH domain, the steering vector of the beamformer is given by the spherical harmonics
stVec = Y_src.T # d_n

# Power of the sources
power_s = np.diag([1, 1, 1]) # We assume all sources have the same power
# Power of the noise (diffuse noise)
power_noise = 1

# Covariance matrix (N_harm x N_harm)
cov_matrix = stVec @ power_s @ stVec.T + power_noise * np.eye(N_harm) / (4 * np.pi)
#  The first term captures how signals are correlated in the SH domain.
# The second term corresponds to diffuse noise. Is modeled as an isotropic random process
# that affects all sensors equally. "4pi" represents the normalization of the power within a sphere.

plt.figure()
plt.imshow(cov_matrix, cmap='viridis')
plt.colorbar()
plt.title("Spherical covariance matrix")
plt.show()

# Weights calculation
w = sphericalBF.SH_MVDR_weights(N_harm, stVec, el_s, az_s, cov_matrix)

# %% BF visialization

# Example src_xyz for plotting 
src_xyz = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# Resolution of the grid
aziRes = 5
elRes = 5

SH_visualization.plotSphFunctionCoeffs(w[:,0], SH_order, 5, 5)
SH_visualization.plotSphFunctionCoeffs(w[:,1], SH_order, 5, 5)
SH_visualization.plotSphFunctionCoeffs(w[:,2], SH_order, 5, 5)