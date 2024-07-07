# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 18:23:50 2024

@author: sergiozc
@references:
    - [1] N. Epain and C. T. Jin, "Spherical Harmonic Signal Covariance and Sound Field Diffuseness"
    - [2] B. Rafaely, Y. Peled, M. Agmon, D. Khaykin, and E. Fisher, "Spherical Microphone Array Beamforming"
    - [3] https://github.com/polarch/Spherical-Array-Processing
    - [4] J. Meyer and G. W. Elko, "Handling Spatial Aliasing in Spherical Array Applications"

"""

import numpy as np
from utils import utils
import matplotlib.pyplot as plt
from SH_visualize import SH_visualization
from SH_BF import sphericalBF
from sphericalHarmonic import SHutils

plt.close('all')

# %% Parameter definition

SH_order = 3 # Spherical harmonic order (it determines the complexity of the function)
# The higher is the order, the more directive is the beampattern
# Nevertheless, the higuer is the order, the  more spatial aliasing might we have

Nmin = 0 # Lowest order (default 0)
N_harm = (SH_order + 1) ** 2 # Highest number of harmonics (not valid if Nmin != 0)

L = 4 # Number of sources
az_s, el_s = sphericalBF.sph_sources(L, 'random') # Sources DOA


# Generar ángulos de elevación aleatorios entre 0 y π/2 radianes
el_s = np.random.uniform(0, np.pi / 2, L)

# Generar ángulos azimutales aleatorios entre 0 y 2π radianes
az_s = np.random.uniform(0, 2 * np.pi, L)


# %% Check the spatial aliasing with b_n.
# See Fig.2 from [4]
# f = 2000
# c = 342
# r = 0.042
# kr = utils.getK(f, c) * r
# b_n = 20*np.log10(SHutils.sph_bn(3, kr))




#%% Beamformer

# Spherical harmonics
Y_s = SHutils.realSH(SH_order, utils.el2inc(el_s), az_s) # d_n (steering vector)
# In SH domain, the steering vector of the beamformer is given by the spherical harmonics

# Power of the sources
power_s = np.eye(L) # We assume all sources have the same power

potencias = np.array([1, 3, 5, 1])
# Crea la matriz diagonal con las potencias
power_s = np.diag(potencias)

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

# Weights calculation
w = sphericalBF.SH_MVDR_weights(N_harm, Y_s, cov_matrix)

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
        
#%% Additional visualizations

#-----------------------------------------------------------------------------
# SPHERICAL HARMONICS
# ¡WARNING!: Many images might be generated
#-----------------------------------------------------------------------------
# # Creating the grid
# theta = np.linspace(0, np.pi, 100)  # Elevation
# phi = np.linspace(0, 2 * np.pi, 100)  # Azimut
# theta, phi = np.meshgrid(theta, phi)

# # Visualization per n and m
# for n_h in range(Nmin, SH_order+1):
#       for m_h in range(-n_h, n_h + 1):
#           SH_visualization.harmonic_plot(SH_order, n_h, m_h, theta, phi)


#-----------------------------------------------------------------------------
# MULTIPLE COVARIANCE MATRIX
#-----------------------------------------------------------------------------
order = 3
N_harm = (order + 1) ** 2
power_noise = 1

# FIRST MATRIX
L1 = 1 # Number of sources
az_s1, el_s1 = sphericalBF.sph_sources(L1) # Sources DOA
Y_s1 = SHutils.realSH(order, utils.el2inc(el_s1), az_s1) #SH
power_s1 = np.eye(L1) 
cov_matrix1 = sphericalBF.SH_COV(N_harm, Y_s1, power_s1, power_noise)

# SECOND MATRIX
L2 = 5 # Number of sources
az_s2, el_s2 = sphericalBF.sph_sources(L2) # Sources DOA
Y_s2 = SHutils.realSH(order, utils.el2inc(el_s2), az_s2) #SH
power_s2 = np.eye(L2) 
cov_matrix2 = sphericalBF.SH_COV(N_harm, Y_s2, power_s2, power_noise)

# THIRD MATRIX
L3 = 20 # Number of sources
az_s3, el_s3 = sphericalBF.sph_sources(L3) # Sources DOA
Y_s3 = SHutils.realSH(order, utils.el2inc(el_s3), az_s3) #SH
power_s3 = np.eye(L3) 
cov_matrix3 = sphericalBF.SH_COV(N_harm, Y_s3, power_s3, power_noise)

# Creating subplot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(cov_matrix1, cmap='viridis')
axes[0].set_title('Covariance Matrix, L = 1')
axes[0].set_xlabel('SH Signal Index')
axes[0].set_ylabel('SH Signal Index')
axes[0].figure.colorbar(axes[0].images[0], ax=axes[0])

axes[1].imshow(cov_matrix2, cmap='viridis')
axes[1].set_title('Covariance Matrix, L = 5')
axes[1].set_xlabel('SH Signal Index')
axes[1].set_ylabel('SH Signal Index')
axes[1].figure.colorbar(axes[1].images[0], ax=axes[1])

axes[2].imshow(cov_matrix3, cmap='viridis')
axes[2].set_title('Covariance Matrix, L = 20')
axes[2].set_xlabel('SH Signal Index')
axes[2].set_ylabel('SH Signal Index')
axes[2].figure.colorbar(axes[2].images[0], ax=axes[2])

plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------------
# COVARIANCE MATRIX MISMATCH VS. NUMBER OF SOURCES VS. SH ORDER
#-----------------------------------------------------------------------------
L_max = 30
order_max = 4
power_noise = 1
dif_mis_p = np.zeros((order_max, L_max))


for order_p in range(1, order_max + 1):
    N_harm_p = (order_p + 1) ** 2
    for L_p in range(1, L_max + 1):
        az_s_p, el_s_p = sphericalBF.sph_sources(L_p)  # Sources DOA
        Y_s_p = SHutils.realSH(order_p, utils.el2inc(el_s_p), az_s_p)  # SH
        power_s_p = np.eye(L_p) 
        cov_matrix_p = sphericalBF.SH_COV(N_harm_p, Y_s_p, power_s_p, power_noise)
        dif_mis_p[order_p - 1, L_p - 1] = sphericalBF.diffuse_mismatch(order_p, L_p, cov_matrix_p)
        
plt.figure()
for order_p in range(order_max):
    plt.plot(range(1, L_max + 1), dif_mis_p[order_p, :], label='SH order = ' + str(order_p + 1))
    plt.scatter(range(1, L_max + 1), dif_mis_p[order_p, :])
# Etiquetas y título
plt.xlabel('Number of Sources (L)')
plt.ylabel(r'Mismatch Value ($\xi$)')
plt.title('Transition to a diffuse sound field')
plt.legend(title='Harmonic Order')
plt.grid(True)
plt.show()

        
