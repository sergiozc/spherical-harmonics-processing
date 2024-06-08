# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 18:01:42 2024

@author: sergiozc
@references: Fahim, Abdullah, et al. “PSD Estimation and Source Separation in a Noisy Reverberant Environment Using a Spherical Microphone Array.” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 26, no. 9, Institute of Electrical and Electronics Engineers (IEEE), Sept. 2018, pp. 1594–607, doi:10.1109/taslp.2018.2835723.

SPHERICAL HARMONIC DECOMPOSITION
"""

import numpy as np
from scipy.special import lpmv, factorial
from math import pi
from bessel_hankel import BHfunctions
import math
from scipy.linalg import pinv

class SHutils:
    
    @staticmethod 
    def getK(freq, c):
        
        """
        Calculate wave number.

        Parameters:
            freq (float or numpy.array): Frequency.
            c (float, optional): Speed of sound. Defaults to 343 m/s.

        Returns:
            float or nump: Wave number.
        """
        
        k = 2 * np.pi * freq / c
        
        return k
    
    @staticmethod 
    def realSHPerMode(n, m, el, az):
        """
        (checked)
        Calculate real spherical harmonics for a specific mode.
    
        Parameters:
            n (int): Order of the spherical harmonic.
            m (int): Mode of the spherical harmonic.
            el (numpy.ndarray): Vector of elevation angles in radians (Qx1).
            az (numpy.ndarray): Vector of azimuth angles in radians (Qx1).
    
        Returns:
            numpy.ndarray: Real spherical harmonics values for the specified mode (Qx1).
        """
        
        cos_el = np.cos(el)
        # Extracting associated Legendre functions of integer order and real degree
        # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lpmv.html
        L = np.zeros((n+1, len(cos_el)))
        for i in range(n+1):
            L[i, :] = lpmv(i, n, cos_el)

        L = L[abs(m) , :]
        
        # Normalization factor
        Norm = (-1) ** m * np.sqrt((2 * n + 1) * factorial(n - abs(m)) / (4 * pi * factorial(n + abs(m))))
    
        if m < 0:
            K = np.sqrt(2) * np.sin(abs(m) * az)
        elif m > 0:
            K = np.sqrt(2) * np.cos(m * az)
        else:
            K = 1
        
        # Real spherical harmonics
        y = Norm * L * K
    
        return y
    
    @staticmethod 
    def complexSHPerMode(n, m, el, az):
        """
        (checked)
        Calculate complex spherical harmonics for a specific mode.
    
        Parameters:
            n (int): Order of the spherical harmonic.
            m (int): Mode of the spherical harmonic.
            el (numpy.ndarray): Vector of elevation angles in radians (Qx1).
            az (numpy.ndarray): Vector of azimuth angles in radians (Qx1).
    
        Returns:
            numpy.ndarray: Complex spherical harmonics values for the specified mode (Qx1).
        """
        
        M = abs(m)
        
        if M > n:
            return 0
        
        m_index = M
        a1 = ((2*n + 1) / (4 * np.pi))
        a2 = factorial(n - M) / factorial(n + M)
        C = np.sqrt(a1 * a2)
        
        cos_el = np.cos(el)
        # Extracting associated Legendre functions of integer order and real degree
        # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lpmv.html
        L = np.zeros((n+1, len(cos_el)))
        for i in range(n+1):
            L[i, :] = lpmv(i, n, cos_el)
        
        y = C * L[m_index, :].T * np.exp(1j * M * az)
        
        if m < 0:
            y = (-1) ** M * np.conj(y)
        
        
        return y
    
    @staticmethod 
    def complexSH(order, el, az):
        """
        (checked)
        Calculate complex spherical harmonics upto an order 
        (first row --> n = 0, m = -n ; last row --> n = order, m = order)
        
    
        Parameters:
            order (int): Highest SH order to calculate
            el (numpy.ndarray): Vector of elevation angles in radians (Qx1).
            az (numpy.ndarray): Vector of azimuth angles in radians (Qx1).
    
        Returns:
            numpy.ndarray: Complex spherical harmonics values (order+1)^2 x Q
        """
        # Check the number of elements of el and az (must be the same)
        if el.size != az.size:
            if el.size == 1:
                el = np.repeat(el, az.size)
            elif az.size == 1:
                    az = np.repeat(az, el.size)
            else:
                raise ValueError('Number mismatch between azimuths and elevations.')
                
        NN = (order + 1)**2
        Q = len(el)
        cos_el = np.cos(el)
        
        y = np.zeros((NN, Q), dtype=np.complex128)
        y[0, :] = 1 / np.sqrt(4 * np.pi)
        
        if order > 0:
            ind = 1
            for n in range(1, order+1):
                L = np.zeros((n + 1, Q))  # Initialize L with zeros
                for i in range(n+1):
                    L[i, :] = lpmv(i, n, cos_el)
                
                # Select all rows except the first one, invert them and concatenate them with the original.
                L = np.vstack([np.flipud(L[1:, :]), L])
                
                m = np.arange(-n, n + 1)
                K = np.exp(1j * np.outer(m, az.T))
                M = abs(m)
                # Normalization factor
                Norm = (2 * n + 1) / (4 * np.pi) * factorial(n-M) / factorial(n+M)
                
                nextind = (n+1)**2
                fact1 = np.where(m < 0, (-1) ** np.abs(m), 1)
                fact2 = np.sqrt(Norm[:, np.newaxis]) * (L * K)
                
                
                y[ind:nextind, :] = fact1[:, np.newaxis] * fact2


                ind = nextind
                
    
        return y
    
    
    @staticmethod 
    def ACNOrderArray(order):
        """
        Get all SH orders in an array according to ACN (Ambisonics Channel Numbering)

        Parameters:
            order (int): Maximum order of the spherical harmonics.

        Returns:
            n_array (numnpy.ndarray): Int array containing all SH orders according to ACN.
        """
        
        n_arr = np.zeros((order + 1) ** 2, dtype=int)
        ind = 1
        for n in range(1, order + 1):
            num_modes = 2 * n
            n_arr[ind:ind + num_modes + 1] = n
            ind += num_modes + 1
        return n_arr
    
    @staticmethod
    def ACNOrderModeArray(order):
        
        """
        Get all SH orders and modes in an array according to ACN
        NM = (order+1)^2 x 2
        """
        
        nm_array = np.zeros(((order+1)**2, 2), dtype=int)
        
        ind = 1
        for n in range(1, order + 1):
            num_modes = 2 * n
            nm_array[ind:ind + num_modes + 1, 0] = n
            nm_array[ind:ind + num_modes + 1, 1] = np.arange(-n, n + 1)
            
            ind += num_modes + 1
            
        return nm_array
    
    
    @staticmethod
    def sph_bn(n_arr, x_arr, is_rigid=True):
        """
        Calculate the coefficients bn for spherical harmonics. Eq (11) from the paper.
    
        Parameters:
            n_arr (numpy.ndarray): Array of orders.
            x_arr (numpy.ndarray): Array of kr values.
            is_rigid (bool): Flag indicating whether to calculate rigid or open boundary coefficients.
    
        Returns:
            numpy.ndarray: Array of bn coefficients.
        """
        
        n_arr = np.array(n_arr).reshape(-1, 1)
        x_arr = np.array(x_arr).reshape(1, -1)
        
        N = len(n_arr)
        K = len(x_arr)
        
        if N > 1:
            # Replicate the x_arr array N times along the 0 axis and 1 time along the 1 axis.
            x_arr = np.tile(x_arr, (N, 1))
    
        if K > 1:
            n_arr = np.tile(n_arr, (1, K))
            
        temp = 0
        
        # Eq (11) from the paper
        if is_rigid:
            temp = BHfunctions.dsph_besselj(n_arr, x_arr) * BHfunctions.sph_hankel1(n_arr, x_arr) / BHfunctions.dsph_hankel1(n_arr, x_arr)
            temp[np.isnan(temp)] = 0
        
        bn = BHfunctions.sph_besselj(n_arr, x_arr) - temp
    
        return bn
    
    @staticmethod
    def pinvRegularized(A, b, weight=0.01):
        """
        Solves Ax = b for x with regularization.
    
        Parameters:
            A (numpy.ndarray): Matrix A of shape (M, N).
            b (numpy.ndarray): Vector b of shape (M, K).
            weight (float, optional): Regularization weight. Default is 0.01.
    
        Returns:
            numpy.ndarray: Solution vector x of shape (N, K).
        """
        # Regularization weight
        w = weight * np.eye(A.shape[1])
    
        # Augmented matrix
        A_aug = np.vstack([A, w])
    
        # Augmented vector
        b_aug = np.pad(b, ((0, w.shape[0]), (0, 0)), mode='constant', constant_values=0)
    
        # Calculate pseudoinverse and solve
        x = pinv(A_aug) @ b_aug
    
        return x
    
    
    
    
    @staticmethod 
    def alphaOmni(P, order, k, r, el, az, isRigid, mode, compensateBesselZero, applyRegularization, weight):
        """
        (checked. The result is not exactly the same due to so many decimals)
        It estimates the sound field coefficients.
        (first row --> n = 0, m = 0 ; last row --> n = order, m = order)
        
        Dependencies:
            BHfunctions
        Parameters:
            P (Q x timeFrames matrix): Sound pressure
            order (int): Highest SH order to calculate
            k (float): Wavenumber
            r (numpy.ndarray) Vector of array radius
            el (numpy.ndarray): Vector of elevation angles in radians (Qx1).
            az (numpy.ndarray): Vector of azimuth angles in radians (Qx1).
            isRigid (boolean): Array type (default 0)
            mode (list): inv/mm (pseudoinverse or mode matching)
            compensateBesselZero: Compensate Bessel zero (used if mode ~= inv), ref: https://ieeexplore.ieee.org/document/8357900
            applyRegularization: Apply regularization (used if mode == inv)
            weight(numpy.ndarray): Array Weight (used if mode ~= inv)
    
        Returns:
            % alpha: Sound field coefficients, (N+1)^2 x T (where variable T represents timeframes)
        """
        
        # kr product
        kr = k * r
        
        # True order of the sound field
        if isinstance(kr, np.ndarray): # Checking if it is an array
            N_true = max(order, math.ceil(kr.max()))
        else:
            N_true = max(order, math.ceil(kr))
        
        # SH orders
        n_arr = SHutils.ACNOrderArray(N_true)
        # Beta
        bn = SHutils.sph_bn(n_arr, kr, isRigid)
        
        if mode == 'inv':
            Y_mat = SHutils.complexSH(N_true, el, az) * bn
            
            if applyRegularization:
                # Apply regularization if necessary
                alpha = SHutils.pinvRegularized(Y_mat.T @ P)
            else:
                alpha = pinv(Y_mat.T) @ P
            
            if order < N_true:
                alpha = alpha[:((order + 1) ** 2), :]
        else:
            #  It is enough to use N for mode-matching case
            bn = bn[:((order+1)**2), :]
            
            # Fixed flooring on bn
            if compensateBesselZero:
                bn = np.maximum(np.abs(bn), 0.05) * np.exp(1j * np.angle(bn))
            
            # NN x Q
            Y_mat = np.conj(SHutils.complexSH(order, el, az)) / bn
            
            if weight is not None and not np.all(weight == 1) and not np.all(weight is None):
                weight = weight.T
                Y_mat = Y_mat * weight
            
            alpha = Y_mat @ P
            
                        
        if isinstance(kr, np.ndarray): # Checking if it is an array
            if (max(np.ceil(kr)) + 1) ** 2 > len(az):
                print('(N + 1)^2 > Q, spatial alising might occur.')
        else:
            if (np.ceil(kr) + 1) ** 2 > len(az):
                print('(N + 1)^2 > Q, spatial alising might occur.')

    
        return alpha 
    
