# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:07:04 2024

@author: sergiozc

Spherical harmonics beamformer methods
"""

import numpy as np
from scipy.linalg import solve

class sphericalBF:
    
    @staticmethod
    def sph_sources(Ns):
        
        """
        Generates sources positions within a sphere
        
        Parameters:
            Ns (int): Number of desired sources
            
        Returns:
            tuple (az_s, el_s): Azimuth and elevation of each source (rad)
        """
        
        index = np.arange(0, Ns, dtype=float) + 0.5
        
        phi = np.arccos(1 - 2*index/Ns)
        theta = np.pi * (1 + 5**0.5) * index
    
        az_s = theta % (2 * np.pi)  # Azimuth
        el_s = phi - np.pi / 2      # Elevation
    
        return az_s, el_s
    
    @staticmethod
    def SH_COV(N_harm, Y, P_s, P_v):
        
        """
        Calculates the SH signal covariance matrix.
        Equation (11) from [1].
        
        Parameters:
            N_harm(int): Highest number of harmonics (determines the size of the matrix)
            Y(numpy.ndarray): Spherical harmonics
            P_s(numpy.ndarray): Power of the sources
            P_v(numpy.ndarray): Power of the noise
            
        Returns:
            np.ndarray: Covariance matrix (N_harm x N_harm)
        """
        
        cov_matrix = Y @ P_s @ Y.T + P_v * np.eye(N_harm) / (4 * np.pi)
        #  The first term captures how signals are correlated in the SH domain.
        # The second term corresponds to diffuse noise. Is modeled as an isotropic random process
        # that affects all sensors equally. "4pi" represents the normalization of the power within a sphere.
        
        return cov_matrix
    
    def diffuse_mismatch(SH_order, L, cov_matrix):
        
        """
        Calculates the mismatch between the SH covariance matrix and the covariance matrix from
        a perfect diffuse sound field.
        Equation (10) from [1].
        
        Parameters:
            SH_order(int): Spherical harmonic order
            L(int): Number of sources (plane waves)
            cov_matrix(numpy.ndarray): SH signal covariance matrix
            
        Returns:
            float: Mismatch
        """
        
        term1 = 1 / ((L+1)**2)
        term2 = cov_matrix / np.linalg.norm(cov_matrix, ord=2) - np.eye((SH_order+1)**2)
        
        xi = term1 * np.linalg.norm(term2, ord='fro')**2

        
        return xi

    @staticmethod
    def SH_MVDR_weights(N_harm, dn, cov_matrix):
        
        """
        Calculates MVDR weights within the SH domain
        
        Parameters:
            N_harm(int): Highest number of harmonics
            dn(numpy.ndarray): Steering vector of the beamformer
            cov_matrix(numpy.ndarray): Covariance matrix
            
        Returns:
            np.ndarray: Matrix which contains the weights of the BF for each source
        """
        
        # Number of beams (sources)
        nBeams = len(dn[1,:])
        # MVDR weights matrix
        w = np.zeros((N_harm, nBeams))
                  
        # Calculating weights for each direction of the beamformer
        for nb in range(nBeams):
            dn_s = dn[:, nb] # Steering vector for each source
            invA_b = solve(cov_matrix, dn_s) # Solving the system: cov_matrix * invA_b = dn_s
            b_invA_b = dn_s @ invA_b # Dot product (like a sum)
            w[:, nb] = invA_b / b_invA_b # MVDR weights
        
        return w
    
    
