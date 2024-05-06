# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:46:35 2024

@author: sergiozc

@references: Fahim, Abdullah, et al. “PSD Estimation and Source Separation in a Noisy Reverberant Environment Using a Spherical Microphone Array.” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 26, no. 9, Institute of Electrical and Electronics Engineers (IEEE), Sept. 2018, pp. 1594–607, doi:10.1109/taslp.2018.2835723.
"""

import numpy as np
from sphericalHarmonic import SHutils
from sympy.physics.wigner import gaunt
from bessel_hankel import BHfunctions
from utils import utils
from scipy.linalg import pinv

class PSDestimation:
    
    @staticmethod
    def c_factor(n, n_p):
        """
        Calculates C factor of the PSD estimation algorithm.
    
        Parameters:
            n (int): Index
            n_p (int): Different index

    
        Returns: 
            C (complex)
            
        """
        C = 16 * (np.pi**2) * 1j**(n - n_p)    
        return C
    
    @staticmethod
    def w_factor(n, n_p, m, m_p, u, v):
        """
        Calculates W factor of the PSD estimation algorithm. Eq (64) from the paper.
    
        Parameters:
            n, m, n_p, m_p (int): Spherical harmonics indexes.
            u (int): Reverberant power order
            v (int): Reverberant power degree

    
        Returns: 
            W (float) 
            
        """
        # Gaunt coefficient --> gaunt()
        # Reference:  https://docs.sympy.org/latest/modules/physics/wigner.html#sympy.physics.wigner.wigner_3j
        W_sym = (-1)**m * gaunt(v, n, n_p, u, -m, m_p) # Eq (64) from de paper
        # Check if the object is an integer or 0. Else --> 64 digits of precision
        if isinstance(W_sym, int) or W_sym == 0:
            W = float(W_sym)
        else:
            W = float(W_sym.n(64))
        

        return W
    
    
    @staticmethod
    def upsilon_term(n, m, n_p, m_p, el_s, az_s):
        """
        Calculates upsilon term of the PSD estimation algorithm. Eq (38) from the paper.
    
        Parameters:
            n, m, n_p, m_p (int): Spherical harmonics indexes.
            el_s (numpy.ndarray): Vector of sources elevation angles in radians (Lx1).
            az_s (numpy.ndarray): Vector of sources azimuth angles in radians (Lx1).

    
        Returns: 
            numpy.ndarray: Complex upsilon term Lx1
            
        """
        
        SH = SHutils.complexSHPerMode(n_p, m_p, el_s, az_s)
        SH_conj = np.conj(SHutils.complexSHPerMode(n, m, el_s, az_s))
        
        ups = PSDestimation.c_factor(n, n_p) * SH_conj * SH
        
    
        return ups
    
    @staticmethod
    def psi_term(n, m, n_p, m_p, u, v):
        """
        Calculates psi term of the PSD estimation algorithm. Eq (39) from the paper.
    
        Parameters:
            n, m, n_p, m_p (int): Spherical harmonics indexes.
            u (int): Reverberant power order
            v (int): Reverberant power degree
    
        Returns:
            numpy.ndarray: Complex psi term 
            
        """
    
        
        psi = PSDestimation.c_factor(n, n_p) * PSDestimation.w_factor(n, n_p, m, m_p, u, v)
    
        return psi
    
    @staticmethod
    def omega_term(n, m, n_p, m_p, k, r):
        """
        Calculates omega term of the PSD estimation algorithm. Eq (42) from the paper.
        It depends on bessel functions and bn, W factor
    
        Parameters:
            n, m, n_p, m_p (int): Spherical harmonics indexes.
            k (float): Wavenumber
            r (numpy.ndarray) Vector of array radius
    
        Returns:
            Omega (array of complex or complex)
            
        """
        kr = k * r
        fact1 = (4 * np.pi)**(3/2) * 1j**(n - n_p + 2*m + 2*m_p)
        fact2 = BHfunctions.sph_besselj(n, kr) * BHfunctions.sph_besselj(n_p, kr)
        fact3 = PSDestimation.w_factor(n, n_p, -m, -m_p, 0, 0)
        
        if isinstance(kr, np.ndarray):
            denom = (np.abs(SHutils.sph_bn(n, kr, True))**2).T
            omega = (fact1 * fact2 * fact3) / denom
        else:
            denom = (np.abs(SHutils.sph_bn(n, kr, True))**2)
            omega = ((fact1 * fact2 * fact3) / denom)[0][0] # [0][0] to return a scalar (not an array)
            
        
    
        return omega
    
    @staticmethod
    def translation_matrix(el_s, az_s, k_array, r, N, V, L):
        """
        Fills the translation matrix with the corresponding terms. See Eq (47) from the paper.
    
        Parameters:
            el_s (numpy.ndarray): Vector of sources elevation angles in radians (Lx1).
            az_s (numpy.ndarray): Vector of sources azimuth angles in radians (Lx1).
            k_array (numpy.ndarray): Wavenumber array. len(k_array) = Nfreq
            r (numpy.ndarray) Vector of array radius
            N (int): Maximun SH order
            V (int): Maximum reverberation power order
            L (int): Number of sources

    
        Returns:
            numpy.tensor: [Nfreq x (N + 1)^4 x (L + (V+1)^2 + 1)]
            
        """
        # Matrix initialization
        rows = (N + 1)**4
        columns = L + (V + 1)**2 + 1
        Nfreq = len(k_array)
        
        # Defining T_matrix as a tensor
        T_matrix = np.zeros((Nfreq, rows, columns), dtype=complex)
        
        ups = np.zeros((rows, L),  dtype=complex)
        omega = np.zeros((rows, Nfreq),  dtype=complex)
        psi = np.zeros((rows, (V+1)**2), dtype=complex)
        
        i = 0
        # n iteration
        for n in range(N+1):
            # m iteration
            for m in range(-n, n+1):
                # n' iteration
                for n_p in range(N+1):
                    # m' iteration
                    for m_p in range(-n_p, n_p+1):
                        upsilon = PSDestimation.upsilon_term(n, m, n_p, m_p, el_s, az_s)
                        ups[i, :] = upsilon                                     
                        # v iteration
                        j = 0
                        for v in range(V+1):
                            # u iteration
                            for u in range(-v, v+1):
                                psi_term = PSDestimation.psi_term(n, m, n_p, m_p, v, u)
                                psi[i, j] = psi_term
                                j += 1
                        
                        # Frequency loop
                        for k_index, k in enumerate(k_array):
                            omega_term = PSDestimation.omega_term(n, m, n_p, m_p, k, r)
                            omega[i, k_index] = omega_term[0]
                        i += 1
        
        # Filling the matrix
        for k in range (Nfreq):
            # Final matrix (translation matrix)
            T_matrix[k, :, :] = np.concatenate((ups, psi, omega[:, k].reshape(-1, 1)), axis=1)
                
        return T_matrix
    
    @staticmethod
    def lambda_matrix(k_array, N, beta, alpha, timeFrames):
        """
        Calculates the expected value of LAMBDA. See Eq (58) from the paper.
        An exponentially weighted moving average algorithm is used (EWMA).
    
        Parameters:
            k_array (numpy.ndarray): Wavenumber
            N (int): Highest SH order
            beta (float): Smoothing factor. In range [0,1].
            alpha (numpy.ndarray): Sound field coefficients, (N+1)^2 x T (where variable T represents timeframes)
            timeFrames (int): Number of time frames
    
        Returns: 
            numpy.ndarray: [T x (N+1)^4] (where variable T represents timeframes)
            
        """
        
        Nfreq = len(k_array)
        # Column matrix
        lambda_matrix = np.zeros((Nfreq, (N+1)**4, timeFrames), dtype=complex)

        
        i = 0 # Index to fill the final matrix (until (N+1)^4)
        j_ = 0 # Index to manage (select rows) alpha values because alpha contains (N + 1)^2 rows (all combinations of n and m)
        for n in range(N+1):
            # m iteration
            for m in range(-n, n+1):
                # n' iteration
                j_p = 0 # Index to manage (select rows) alpha' values
                for n_p in range(N+1):
                    # m' iteration
                    for m_p in range(-n_p, n_p+1):
                        for k_index in range(Nfreq):
                            data = alpha[k_index, j_, :] * np.conj(alpha[k_index, j_p, :])
                            lambda_matrix[k_index, i, :] = utils.ewma_tensor(beta, data)
                        j_p += 1
                        i += 1
                j_ += 1
        
        return lambda_matrix
    
    
    @staticmethod
    def psd_matrix(T_matrix, lambda_matrix):
        """
        Calculates the expected value of estimated source and noise PSDs at the origin.
        See Eq (49) from the paper.
    
        Parameters:
            T_matrix (numpy.ndarray): [Nfreq x (N + 1)^4 x (L + (V+1)^2 + 1)]
            lambda_matrix (numpy.ndarray): [Nfreq x (N + 1)^4 x 1]
    
        Returns: 
            numpy.ndarray: [Nfreq x (L + (V + 1)**2 + 1)]
        """
        # Transpose lambda_matrix to match the dimensions for multiplication
        lambda_matrix = np.expand_dims(lambda_matrix, axis=-1)
    
        # Solve the system for each frequency
        theta = np.linalg.pinv(T_matrix) @ lambda_matrix
        
        # Delete last dimension 
        theta = theta[:, :, 0]
        
        
     
        return theta
    