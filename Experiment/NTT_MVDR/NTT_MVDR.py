# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 16:46:08 2024

@author: sergiozc
@references: 
    - Higuchi, Takuya, et al. "Online MVDR beamformer based on complex Gaussian mixture model with spatial prior for noise robust ASR."
    
Online MVDR Beamformer Based on Complex Gaussian Mixture Model With Spatial Prior for Noise Robust ASR.
"""

import numpy as np
from scipy.signal import get_window
from scipy.linalg import eigh, toeplitz, inv, det

class NTT_BF:
    
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
    def stft_multi_NTT(x, fs=16000):
        wlen = 400  # 25 ms a 16 kHz
        win = np.sqrt(get_window('hann', wlen, fftbins=True))
        nsampl, nchan = x.shape
        nbin = (wlen // 2) + 1
        desp = int(0.25 * wlen)  # Solapamiento del 75%
        nfram = int(np.ceil((nsampl - wlen) / desp)) + 1
        
        # Rellenamos con ceros just in case
        dif = wlen + (nfram - 1) * desp - nsampl
        x = np.vstack([x, np.zeros((dif, nchan))])
        X = np.zeros((nbin, nfram, nchan), dtype=np.complex64)
    
        for c in range(nchan):
            ind = 0
            for t in range(0, dif + nsampl - wlen + 1, desp):
                frame = x[t:t + wlen, c] * win
                transf = np.fft.fft(frame, n=wlen)
                X[:, ind, c] = transf[:nbin]
                ind += 1
    
        return X
    
    @staticmethod
    def istft_multi_NTT(X, nsampl):
        nbin, nfram, nchan = X.shape
        wlen = 2 * (nbin - 1)
        win = np.sqrt(get_window('hann', wlen, fftbins=True))
        desp = int(0.25 * wlen)  # Solapamiento del 75%
        totals = (nfram - 1) * desp + wlen
        x = np.zeros((totals, nchan))
    
        for c in range(nchan):
            for t in range(nfram):
                frame = X[:, t, c]
                frame = np.concatenate([frame, np.conj(frame[-2:0:-1])])
                frame = np.fft.ifft(frame)
                x[(t * desp):(t * desp + wlen), c] += win * np.real(frame)
    
        return x[:nsampl, :]
    
    @staticmethod
    def IniBatch(Y):
        nbin, nfram, nchan = Y.shape
        Cor = {'y': [], 'n': []}
    
        for f in range(nbin):
            Ytmp = np.transpose(Y[f, :, :], (1, 0))
            Cor['y'].append(np.cov(Ytmp))
            Cor['n'].append(np.eye(nchan))
    
        return Cor
    
    @staticmethod
    def evMultGauss_Complex(x, iS, detS):
        """
        Evaluates the multivariate Gaussian probability density function for complex numbers.
    
        Parameters:
        x (numpy.ndarray): The complex input vector.
        iS (numpy.ndarray): The inverse of the covariance matrix.
        detS (float): The determinant of the covariance matrix.
    
        Returns:
        float: The evaluated probability density.
        """
        # Calculate the exponent term
        exponent_term = -np.real(np.dot(np.dot(x.conj().T, iS), x))
        
        # Calculate the probability density
        p = np.abs(np.real(np.exp(exponent_term) / detS))
        
        return p

    @staticmethod
    def ntt_bf(y, Nmic):
        
        # HYPER-PARAMETERS DEFINITION
        vect1 = np.ones((Nmic, 1))
        Niter = 1 # Number of iterations
        pi6 = np.pi**6
        pow_thresh = -20  # Threshold in dB below which a microphone is considered to fail.
        regul = 1e-3  # Regularization factor for MVDR
        
        # Number of samples of the signal
        nsampl, _ = y.shape
        
        # Check if any mic has failed
        ypow = np.sum(y**2, axis=0)
        ypow = 10 * np.log10(ypow / np.max(ypow))
        fail = (ypow <= pow_thresh)
        Nef = Nmic - np.sum(fail)
        
        # Frequency domain
        Y = NTT_BF.stft_multi_NTT(y, fs=16000)
        nbin, nfram, nchan = Y.shape
        
        # Yspec --> 1: no. canal, 2: bin freq., 3: no. trama
        Yspec = np.mean(np.abs(Y)**2, axis=1).T
        
        # Matrix initilization
        X = np.zeros((nbin, nfram))
        Xpost = np.zeros((nbin, nfram))
        Wpost = np.zeros((nbin, nfram))
        
        Cor = NTT_BF.IniBatch(Y)
        
        
        Q = np.zeros((nbin, Niter))
        X = np.zeros((nbin, nfram), dtype=complex)
        Wpost = np.zeros((nbin, nfram))
        
        for k in range(nbin):  # Each frequency
            # R Initialization
            Ry = Cor['y'][k]
            Rn = Cor['n'][k]
            
            for niter in range(Niter):  # EM algorithm
                # Initilizations
                Lamby = 0
                Lambn = 0
                Rtmpy = np.zeros((Nmic, Nmic), dtype=complex)
                Rtmpn = np.zeros((Nmic, Nmic), dtype=complex)
                Ctmpn = np.zeros((Nmic, Nmic), dtype=complex)
                Ry = Ry + 1e-5 * np.eye(Nmic, dtype=complex)
                Rn = Rn + 1e-5 * np.eye(Nmic, dtype=complex)
                detRy = pi6 * det(Ry)
                detRn = pi6 * det(Rn)
                iRy = inv(Ry)
                iRn = inv(Rn)
                
                for t in range(nfram):  # Procesar cada instante
                    ya = Y[k, t, :].reshape(-1, 1)
                    Ya2 = ya @ ya.T
                    
                    phi_y = (1/nchan) * np.sum(np.diag(Ya2 @ iRy))
                    phi_n = (1/nchan) * np.sum(np.diag(Ya2 @ iRn))
    
                    detCy = phi_y**6 * detRy
                    detCn = phi_n**6 * detRn
                    iCy = iRy / phi_y
                    iCn = iRn / phi_n
                    prob_y = NTT_BF.evMultGauss_Complex(ya, iCy, detCy)
                    prob_n = NTT_BF.evMultGauss_Complex(ya, iCn, detCn)
                    lamb_y = prob_y / (prob_y + prob_n)
                    lamb_n = 1 - lamb_y
                    
                    # Accumulate
                    Lamby += lamb_y
                    Lambn += lamb_n
                    Rtmpy += lamb_y * Ya2 / phi_y
                    Rtmpn += lamb_n * Ya2 / phi_n
                    Ctmpn += lamb_n * Ya2
                    
                    Q[k, niter] += lamb_y * np.log(prob_y) + lamb_n * np.log(prob_n)
                    
                Ry = Rtmpy / Lamby
                Rn = Rtmpn / Lambn
                
            Cor['n'][k] = Ctmpn / Lambn
        
            R_x = Cor['y'][k] - Cor['n'][k]
            Dx, Vx = eigh(R_x)
            posm = np.argmax(np.abs(Dx))
            Df = Vx[:, posm]  # Steering vector
            
            # Beamformer
            # MVDR (two phases)
            F = np.diag(1.0 / Df)
            Phinoise = F @ Cor['n'][k] @ F.T + regul * np.diag(Yspec[~fail, k])
            W = inv(Phinoise) @ vect1[~fail] / (vect1[~fail].T @ inv(Phinoise) @ vect1[~fail])
            
            # Postfiltering
            Phi = np.zeros((Nef, Nef), dtype=complex)
            for t in range(10):
                ya = Y[k, t, :].reshape(-1, 1)
                Yin = ya[~fail] / Df[~fail]  # Alinear
                Yout = np.conj(W) * Yin  # Marro
                Phi += Yout @ Yout.T
            Phi /= 10
            T = toeplitz(np.concatenate(([0], np.ones(Nef-1))), np.zeros(Nef))
            
            # Beamforming
            for t in range(nfram):
                ya = Y[k, t, :].reshape(-1, 1)
                # Aplicamos beamforming
                X[k, t] = W.T @ ya[~fail]
                # Procesado de MVDR con postfilter
                Yin = ya[~fail] / Df[~fail]  # Alinear
                Yout = np.conj(W) * Yin
                X[k, t] = np.sum(Yout)
                Phi = 0.9 * Phi + 0.1 * Yout @ Yout.T  # Marro
                factor = np.real(W.T @ W) / np.real(W.T @ T @ W)  # Marro
                Wpost[k, t] = factor * np.real(np.sum(T * Phi)) / np.real(np.sum(np.diag(Phi)))
                Wpost[k, t] = np.clip(Wpost[k, t], 0, 1)
                X[k, t] *= Wpost[k, t]
        
        # ISTFT
        enh = NTT_BF.istft_multi_NTT(X,nsampl)
        
        return R_x, enh