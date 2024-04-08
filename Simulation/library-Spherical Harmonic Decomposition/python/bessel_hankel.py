# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:07:02 2024

@author: sergiozc

@references: Fahim, Abdullah, et al. “PSD Estimation and Source Separation in a Noisy Reverberant Environment Using a Spherical Microphone Array.” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 26, no. 9, Institute of Electrical and Electronics Engineers (IEEE), Sept. 2018, pp. 1594–607, doi:10.1109/taslp.2018.2835723.

BESSEL AND HANKEL FUNCTIONS
"""

import numpy as np
from scipy.special import jv, yv

class BHfunctions:
    
    @staticmethod
    def sph_besselj(n, x):
        """
        (checked)
        Spherical Bessel function of the first kind.
    
        Parameters:
            n (int or array_like): Order of the Bessel function.
            x (float or array_like): Argument at which to evaluate the Bessel function.
    
        Returns:
            numpy.ndarray: Value of the spherical Bessel function of the first kind.
        """
        # Encontrar ceros en el argumento
        idx_zero = (x == 0)
    
        # Evaluar la función Bessel de primera especie
        j = np.sqrt(np.pi / (2 * x)) * jv(n + 0.5, x)
    
        # Manejar casos especiales cuando x es cero
        j = np.where(np.logical_and(n == 0, idx_zero), 1, j)
        j = np.where(np.logical_and(n != 0, idx_zero), 0, j)
    
        return j
    
    @staticmethod
    def dsph_besselj(n, x):
        """
        (checked)
        Spherical Bessel function derivative of the first kind.
        
        Parameters:
            n (int): Order of the Bessel function.
            x (float or array_like): Argument at which to evaluate the derivative of the Bessel function.
        
        Returns:
           numpy.ndarray: Value of the spherical Hankel function of the first kind.
        """
        dj = 1 / (2 * n + 1) * (n * BHfunctions.sph_besselj(n - 1, x) - (n + 1) * BHfunctions.sph_besselj(n + 1, x))
        return dj
    
    @staticmethod
    def sph_bessely(n, x):
        """
        Spherical Bessel function of the second kind.
        
        Parameters:
            n (int or array_like): Order of the Bessel function.
            x (float or array_like): Argument at which to evaluate the Bessel function.
        
        Returns:
            numpy.ndarray: Value of the spherical Bessel function of the second kind.
        """
        
        j = np.sqrt(np.pi / (2 * x)) * yv(n + 0.5, x)
        
        return j
    
    @staticmethod
    def dsph_bessely(n, x):
        """
        Spherical bessel function derivative of the second kind.
        
        Parameters:
            n (int): Order of the Bessel function.
            x (float or array_like): Argument at which to evaluate the derivative of the Bessel function.
        
        Returns:
            numpy.ndarray: Value of the derivative of the spherical Bessel function of the second kind.
        """
    
        dj = 1 / (2 * n + 1) * (n * BHfunctions.sph_bessely(n - 1, x) - (n + 1) * BHfunctions.sph_bessely(n + 1, x))
        return dj
    
    @staticmethod
    def sph_hankel1(n, x):
        """
        (checked)
        Spherical Hankel function of the first kind.
        
        Parameters:
            n (int): Order of the Hankel function.
            x (float or array_like): Argument at which to evaluate the Hankel function.
        
        Returns:
            numpy.ndarray: Value of the derivative of the spherical Bessel function of the first kind.
        """
        j = BHfunctions.sph_besselj(n, x) + 1j * BHfunctions.sph_bessely(n, x)
        return j
    
    
    @staticmethod
    def dsph_hankel1(n, x):
        """
        (checked)
        Spherical hankel function derivative of the first kind.
        
        Parameters:
            n (int): Order of the Hankel function.
            x (float or array_like): Argument at which to evaluate the derivative of the Hankel function.
        
        Returns:
            numpy.ndarray: Value of the derivative of the spherical Hankel function of the first kind.
        """
        
        dj = BHfunctions.dsph_besselj(n, x) + 1j * BHfunctions.dsph_bessely(n, x)
        
        return dj
        