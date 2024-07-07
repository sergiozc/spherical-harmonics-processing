# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 18:28:47 2024

@author: sergiozc

Some useful functions
"""

import numpy as np


class utils:
    
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
    def cart2sph(x, y, z):
        
        """
        Converts cartesian to azimuthal and elevation values.

        Parameters:
            x: X coordenate
            y: Y coordenate
            z: Z coordenate

        Returns:
            float: elevation (rad)
            float: azimuthal (rad)
        """
        
        # Position vector
        r = np.sqrt(x**2 + y**2 + z**2)
        # Calcular la elevación (theta)
        el = np.arccos(z / r)
        # Calcular el azimut (phi)
        az = np.arctan2(y, x)
        
        # Make sure [0, 2pi]
        # az[az < 0] += 2 * np.pi
        
        
        return el, az, r
    
    
    @staticmethod
    def sph2cart(r, theta, phi):
        """
        Converts spherical to cartesian coordenates.

        Parameters:
            r: position vector
            theta: elevation
            phi: azimut

        Returns:
                cartesian coordenates (x, y, z)
        """
        x = r * np.cos(theta) * np.cos(phi)
        y = r * np.cos(theta) * np.sin(phi)
        z = r * np.sin(theta)
        return x, y, z
    
    
    @staticmethod
    def el2inc(el):
        """
        Converts elevation angle to inclination
        
        Parámetros:
        el(numpy.ndarray): Elevation in rad
        
        Returns:
            numpy.ndarray; Inclination in rad
        """
        inc = np.pi / 2 - el
        return inc
    
    
    @staticmethod
    def calculate_rows(Nmin, N):
        """
        Calculates the corresponding number of rows.
    
        Parameters:
           Nmin (int): Minimun SH order.
           N (int): Maximum SH order
    
        Returns:
            int: number of rows
        """
        total = sum(2*n + 1 for n in range(Nmin, N + 1))
        return total ** 2
    
    @staticmethod 
    def ewma(beta, data):
        
        """
        Calculates the Exponentially Weighted Moving Average (EWMA).
    
        Parameters:
            data (numpy.ndarray): input data.
            beta (float): Smoothing factor.
    
        Returns:
            numpy.ndarray: Smoothed expected value EWMA.
        """
        ewma_values = np.zeros_like(data)
        ewma_values[0] = data[0]
    
        for tau in range(1, len(data)):
            ewma_values[tau] = beta * ewma_values[tau - 1] + (1 - beta) * data[tau]
    
        return ewma_values
        

    @staticmethod 
    def ewma_tensor(beta, data):
        """
        Calculates the Exponentially Weighted Moving Average (EWMA) for a 3-dimensional tensor.

        Parameters:
            beta (float): Smoothing factor.
            data (numpy.ndarray): Input tensor.

        Returns:
            numpy.ndarray: Smoothed expected value EWMA.
        """
        ewma_values = np.zeros_like(data)
        
        # Initial values (same as ewma_values[:, :, 0] = data[:, :, 0])
        ewma_values[..., 0] = data[..., 0]

        for tau in range(1, data.shape[-1]):
            ewma_values[..., tau] = beta * ewma_values[..., tau - 1] + (1 - beta) * data[..., tau]

        return ewma_values
        
        