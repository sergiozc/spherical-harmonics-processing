# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 18:28:47 2024

@author: sergiozc
"""

import numpy as np


class utils:
    
    @staticmethod 
    def cart2sph(x, y, z):
        
        """
        Converts cartesian to azimuthal and elevation values.

        Parameters:
            x: X coordenate
            y: Y coordenate
            z: Z coordenate
            center: center of the sphere (x0, y0, z0)

        Returns:
            float: elevation (rad)
            float: azimuthal (rad)
        """
        
        # Calcular la magnitud del vector posición (r)
        r = np.sqrt(x**2 + y**2 + z**2)
        # Calcular la elevación (theta)
        el = np.arccos(z / r)
        # Calcular el azimut (phi)
        az = np.arctan2(y, x)
        
        
        return el, az, r
    
    @staticmethod 
    def ewma(beta, data):
        
        """
        Calculates the Exponentially Weighted Moving Average (EWMA).
    
        Parameters:
            data (numpy.ndarray): input data.
            beta (float): Smoothing factor.
    
        Retorna:
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
        
        # Initial values
        ewma_values[..., 0] = data[..., 0]

        for tau in range(1, data.shape[-1]):
            ewma_values[..., tau] = beta * ewma_values[..., tau - 1] + (1 - beta) * data[..., tau]

        return ewma_values
        
        