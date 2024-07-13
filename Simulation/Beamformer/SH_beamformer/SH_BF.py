# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:07:04 2024

@author: sergiozc

Spherical harmonics beamformer methods.

"""

import numpy as np
from scipy.linalg import solve
from scipy.special import factorial, eval_legendre

class sphericalBF:
    
    @staticmethod
    def sph_sources(Ns, distribution = 'uniform'):
        
        """
        Generates sources positions within a sphere
        
        Parameters:
            Ns (int): Number of desired sources
            distribution (string): Uniform distribution or random
            
        Returns:
            tuple (az_s, el_s): Azimuth and elevation of each source (rad)
        """
        
        if distribution == 'uniform':
            index = np.arange(0, Ns, dtype=float) + 0.5
            
            phi = np.arccos(1 - 2*index/Ns)
            theta = np.pi * (1 + 5**0.5) * index
        
            az_s = theta % (2 * np.pi)  # Azimuth
            el_s = phi - np.pi / 2      # Elevation
        else:
            el_s = np.random.uniform(0, np.pi / 2, Ns)
            az_s = np.random.uniform(0, 2 * np.pi, Ns)
    
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
    
    @staticmethod
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
    
    @staticmethod
    def cheby_coeffs(n):
        
        """
        Calculates the polynomial coefficients of a Chebyshev polinomial of order n
        
        Parameters:
            n (int): Order of the polynomial
            
        Returns:
            np.ndarray: Polynomial coefficients
        """
        
        if n == 0:
            return np.array([1])
        
        coeffs = []
        for k in range(n // 2 + 1):
            coef_k = (n / 2) * (-1)**k * factorial(n - k - 1) / (factorial(k) * factorial(n - 2 * k)) * 2**(n - 2 * k)
            coeffs.append(coef_k)
        
        T = np.zeros(n + 1)
        if n % 2 == 0:
            T[::2] = coeffs[::-1]
        else:
            T[1::2] = coeffs[::-1]
        
        return T
    
    @staticmethod
    def legendre_coeffs(n):
        """
        Calculates the polynomial coefficients of a Legendre polynomial of order n
        
        Parameters:
            n (int): Order of the polynomial
            
        Returns:
            np.ndarray: Polynomial coefficients
        """
        
        coeffs = np.zeros(n + 1)
        for k in range(n // 2 + 1):
            coeff_k = ((-1)**k * factorial(2 * n - 2 * k) / (2**n * factorial(k) * factorial(n - k) * factorial(n - 2 * k)))
            coeffs[2 * k] = coeff_k
        
        return coeffs[::-1]
    
    @staticmethod
    def SH_cheby_weights(N, paramType, arrayParam):
        """
        Calculates the spherical weights for a Dolph-Chebyshev beamformer in the SH domain.
        @reference: A. Koretz and B. Rafaely, “Dolph–chebyshev beampattern design for spherical arrays,” IEEE
        transactions on Signal processing, vol. 57, no. 6, pp. 2417–2420, 2009
        
        Parameters:
            n (int): Order of the polynomial
            
        Returns:
            np.ndarray: Polynomial coefficients
        """
        
        M = 2 * N
    
        if paramType == 'sidelobe':
            R = 1 / arrayParam
            x0 = np.cosh((1 / M) * np.arccosh(R))
        elif paramType == 'mainlobe':
            a0 = arrayParam / 2
            x0 = np.cos(np.pi / (2 * M)) / np.cos(a0 / 2)
            R = np.cosh(M * np.arccosh(x0))
    
        t_2N = sphericalBF.cheby_coeffs(2 * N)
    
        P_N = np.zeros((N + 1, N + 1))
        for n in range(N + 1):
            P_N[:n + 1, n] = sphericalBF.legendre_coeffs(n)
    
        d_n = np.zeros(N + 1)
        for n in range(N + 1):
            temp = 0
            for i in range(n + 1):
                for j in range(N + 1):
                    for m in range(j + 1):
                        temp += (1 - (-1)**(m + i + 1)) / (m + i + 1) * \
                                factorial(j) / (factorial(m) * factorial(j - m)) * \
                                (1 / 2**j) * t_2N[2*j] * P_N[i, n] * x0**(2 * j)
            d_n[n] = (2 * np.pi / R) * temp
    
        norm = np.sum(d_n * np.sqrt(2 * np.arange(N + 1) + 1))
        d_n = np.sqrt(4 * np.pi) * d_n / norm
    
        return d_n
    
