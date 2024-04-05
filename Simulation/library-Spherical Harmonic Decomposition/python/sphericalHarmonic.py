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

class SHutils:
    
    @staticmethod 
    def getK(freq, c):
        
        """
        Calculate wave number.

        Parameters:
            freq (float): Frequency.
            c (float, optional): Speed of sound. Defaults to 343 m/s.

        Returns:
            float: Wave number.
        """
        
        lambd = c / freq
        k = (2 * np.pi) / lambd
        
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
        
        # COMPROBAR los "+ 1"
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
        Calculate complex spherical harmonics upto an order.
    
        Parameters:
            order (int): Highest SH order to calculate
            el (numpy.ndarray): Vector of elevation angles in radians (Qx1).
            az (numpy.ndarray): Vector of azimuth angles in radians (Qx1).
    
        Returns:
            numpy.ndarray: Complex spherical harmonics values for the specified mode (Qx1).
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
    def betaFunction():
        """
        Get all SH orders in an array according to ACN (Ambisonics Channel Numbering)

        Parameters:
            order (int): Maximum order of the spherical harmonics.

        Returns:
            n_array (numnpy.ndarray): Int array containing all SH orders according to ACN.
        """
        
        
        beta = "Not ready yet"

        return beta
    
    
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
    def sph_bn(n_arr, x_arr, is_rigid=False):
        """
        Calculate the coefficients bn for spherical harmonics.
    
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
    
        if is_rigid:
            temp = BHfunctions.dsph_besselj(n_arr, x_arr) * BHfunctions.sph_hankel1(n_arr, x_arr) / BHfunctions.dsph_hankel1(n_arr, x_arr)
            temp[np.isnan(temp)] = 0
        
        bn = BHfunctions.sph_besselj(n_arr, x_arr) - temp
    
        return bn
    
    
    
    
    @staticmethod 
    def alphaOmni(P, order, k, r, el, az, isRigid, mode, compensateBesselZero, applyRegularization, weight):
        """
        (checked)
        Calculate complex spherical harmonics upto an order.
        
        Dependencies:
            complexSH
        Parameters:
            P (Q x timeFrames matrix): Sound pressure
            order (int): Highest SH order to calculate
            k (float): Wavenumber
            r (numpy.ndarray) Vector of array radius
            el (numpy.ndarray): Vector of elevation angles in radians (Qx1).
            az (numpy.ndarray): Vector of azimuth angles in radians (Qx1).
            isRigid (boolean): Array type (default 0)
            mode (list): inv/mm (pseudoinverse or mode matching)
            compensateBesselZero: Compensate Bessel zero (uused if mode ~= inv), ref: https://ieeexplore.ieee.org/document/8357900
            applyRegularization: Apply regularization (used if mode == inv)
            weight(numpy.ndarray): Array Weight (used if mode ~= inv)
    
        Returns:
            % alpha: Sound field coefficients, (N+1)^2 x T
        """
        alpha = 'Not ready yet'
    
        return alpha 
    
                
            
        

