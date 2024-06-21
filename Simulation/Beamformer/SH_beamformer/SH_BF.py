# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:07:04 2024

@author: sergiozc

Spherical harmonics beamformer methods
"""

import numpy as np
from scipy.linalg import solve

class sphericalBF:

    def SH_MVDR_weights(N_harm, dn, el_s, az_s, cov_matrix):
        
        """
        Calculates MVDR weights within the SH domain
        
        Parameters:
            N_harm(int): Highest number of harmonics
            dn(numpy.ndarray): Steering vector of the beamformer
            el_s(numpy.ndarray): Vector of elevation angles in radians
            az_s(numpy.ndarray):  Vector of azimuth angles in radians
            cov_matrix(numpy.ndarray): Covariance matrix
            
        Returns:
            np.ndarray: Matrix which contains the weights of the BF for each source
        """
        
        # Number of beams (sources)
        nBeams = len(el_s)
        # MVDR weights matrix
        w = np.zeros((N_harm, nBeams))
                  
        # Calculating weights for each direction of the beamformer
        for nb in range(nBeams):
            dn_s = dn[:, nb] # Steering vector for each source
            invA_b = solve(cov_matrix, dn_s) # Solving the system: cov_matrix * invA_b = dn_s
            b_invA_b = dn_s @ invA_b # Dot product (like a sum)
            w[:, nb] = invA_b / b_invA_b # MVDR weights
        
        return w
