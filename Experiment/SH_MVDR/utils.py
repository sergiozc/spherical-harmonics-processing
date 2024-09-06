# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 18:28:47 2024

@author: sergiozc

Some useful functions
"""

import numpy as np
from scipy.fft import fft
import soundfile as sf
import os


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
        # Calcular la elevación (phi)
        el = np.arcsin(z / r)
        # Calcular el azimut (theta)
        az = np.arctan2(y, x)
        
        # Make sure [0, 2pi]
        # az[az < 0] += 2 * np.pi
        
        
        return el, az, r
    
    
    @staticmethod
    def sph2cart(r, theta, phi):
        """
        Converts spherical to cartesian coordinates.
    
        Parameters:
            r: position vector (radius)
            theta: azimuthal angle (in radians)
            phi: elevation angle (in radians)
    
        Returns:
            cartesian coordinates (x, y, z)
        """
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.cos(phi) * np.sin(theta)
        z = r * np.sin(phi)
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
    def load_wav_signals(directory, num_channels=19):
        """
        Load multiple WAV files from a directory into a matrix with each column representing a channel.
    
        Parameters:
            directory (str): The path to the directory containing the WAV files.
            num_channels (int): The number of channels (default is 19).
    
        Returns:
            np.ndarray: A 2D array where each column is a signal from a different channel.
            int: The sample rate of the WAV files.
        """
        signals = []
        sample_rate = None
        
        for i in range(num_channels):
            file_path = os.path.join(directory, f'signal_{i+1}.wav')
            signal, sr = sf.read(file_path)
            if sample_rate is None:
                sample_rate = sr
            elif sample_rate != sr:
                raise ValueError(f"Sample rate mismatch: {sample_rate} != {sr} for file {file_path}")
            signals.append(signal)
        
        # Stack signals into a 2D array with shape (num_samples, num_channels)
        channels = np.column_stack(signals)
        
        return channels, sample_rate
    
    
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
    
    @staticmethod
    def dB2linear(dB_value):
        """
        Converts from dB to linear.
        
        Parameters:
        dB_value (float): Value in decibels.
        
        Returns:
        float: Linear value
        """
        return 10**(dB_value / 10.0)
    
    @staticmethod
    def calculates_power(signal):
        """
        Calculates the power of a given signal

        Parameters:
            signal (numpy.ndarray): signal in time domain

        Returns:
            numpy.ndarray: total power of the corresponding signal
        """
    
        # If stereo
        if len(signal.shape) > 1:
            signal = signal[:, 0]
    
        # Normalization
        signal = signal / np.max(np.abs(signal))
    
        # Frequency domain
        signal_f = fft(signal)
        
        # Total power
        power = np.sum(np.abs(signal_f)**2) / len(signal)

        return power
    
    @staticmethod
    def calculates_SNR(power_signal, power_noise):
        """
        Calculates signal-to-noise ratio

        Parameters:
            power_signal (float64): power of the signal
            power_noise (float64): power of the noise

        Returns:
            numpy.ndarray: total power of the corresponding signal
        """
        return 10 * np.log10(power_signal / power_noise)
    
    @staticmethod
    def calculates_SDR(x, xest):
        """
        Calculates signal-to-noise ratio

        Parameters:
            x (numpy.ndarray): original signal
            xest (numpy.ndarray): processed signal

        Returns:
            float64: Signal-to-Distorsion-Ratio
        """
    
        # Check-in
        Ex = np.sum(x*x)
        Exest = np.sum(xest*xest)
        if ((Ex < 1e-3) or (Exest < 1e-3)):
            raise NotImplementedError("Error in SDR: null power signals")
    
        # Compute SDR
        scalefactor = np.sum(xest*x)/Ex
        refactor = 7
        target = scalefactor * x
        residual = target - xest
        EtargetdB = 10*np.log10(np.sum(target*target))
        EresidualdB = 10*np.log10(np.sum(residual*residual)+1e-12)
        SDR = EtargetdB - EresidualdB
    
        return SDR