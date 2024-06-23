# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 12:44:42 2024

@author: sergiozc

Methods to visualize spherical harmonics and functions
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import utils
from sphericalHarmonic import SHutils
from scipy.special import sph_harm

class SH_visualization:
    
    @staticmethod
    def harmonic_plot(order, n, m, el, az):
        
        """
        Visualization of all spherical harmonics

        Parameters:
            order (int): Maximum order of the spherical harmonics.
            n (int): Order of the spherical harmonic.
            m (int): Mode of the spherical harmonic.
            el (numpy.ndarray): Vector of elevation angles in radians.
            az (numpy.ndarray): Vector of azimuth angles in radians.
        """
        
        Y_nm = sph_harm(m, n, az, el)
        
        # Harmonic magnitude
        r = np.abs(Y_nm)
        
        # Cartesian coordenates
        x, y, z = utils.sph2cart(r, el, az)
        
        fig = plt.figure(figsize=(14, 7))
        
        # Real part
        ax1 = fig.add_subplot(121, projection='3d')
        real_values = np.real(Y_nm)
        surf_real = ax1.plot_surface(x, y, z, facecolors=plt.cm.seismic(real_values), rstride=1, cstride=1, linewidth=0, antialiased=False)
        ax1.set_title(f'Real part (n={n}, m={m})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        fig.colorbar(surf_real, ax=ax1, shrink=0.5, aspect=5)
        
        # Imaginary part
        ax2 = fig.add_subplot(122, projection='3d')
        imag_values = np.imag(Y_nm)
        surf_imag = ax2.plot_surface(x, y, z, facecolors=plt.cm.seismic(imag_values), rstride=1, cstride=1, linewidth=0, antialiased=False)
        ax2.set_title(f'Imaginary part (n={n}, m={m})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        fig.colorbar(surf_imag, ax=ax2, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def grid2dirs(aziRes, elRes, ZEROED_OR_CENTERED=1, POLAR_ELEV=1):
        """
        Create vectors of azimuthal and elevation grid points based on azimuthal and elevation resolution.
        (CHECKED)
    
        Parameters:
        aziRes (int): Azimuthal resolution in degrees.
        elRes (int): Elevation resolution in degrees.
        ZEROED_OR_CENTERED (int, optional): If 1, azimuth angles start from 0. If 0, azimuth angles start from -180. Default is 1.
        POLAR_ELEV (int, optional): If 1, inclination angles. If 0, elevation angles.
    
        Returns:
        tuple: Two numpy arrays, azimuth and elevation angles in radians.
        """
        
        if 360 % aziRes != 0 or 180 % elRes != 0:
            raise ValueError('Azimuth or elevation resolution should divide exactly 360 and 180 degrees')
        
        # Phi
        if ZEROED_OR_CENTERED:
            azimuth = np.deg2rad(np.linspace(0, 360 - aziRes, num=72))
        else:
            azimuth = np.deg2rad(np.arange(-180, 180-aziRes, aziRes))
        
        # Theta
        if POLAR_ELEV:
            elevation = np.deg2rad(np.arange(0, 180 + elRes, elRes))
        else:
            elevation = np.deg2rad(np.arange(-90, 90 + elRes, elRes))  # elevation angle
        
        Nphi = len(azimuth)
        Ntheta = len(elevation)
        
        azimuth_grid = [0]
        elevation_grid = [0]
        
        
        for i in range(1, Ntheta-1):
            azimuth_grid.extend(azimuth)
            elevation_grid.extend([elevation[i]] * Nphi)
        
        
        if POLAR_ELEV:
            elevation_grid[0] = 0
            azimuth_grid[0] = 0
            elevation_grid.extend([np.pi])
            azimuth_grid.extend([0])
        else:
            elevation_grid[0] = -np.pi / 2
            azimuth_grid[0] = 0
            elevation_grid.extend([np.pi / 2])
            azimuth_grid.extend([0])
        
        return np.array(elevation_grid), np.array(azimuth_grid)
    
    
    @staticmethod
    def Fdirs2grid(W, aziRes, elRes, CLOSED=0):
        """
        Replicate vector function values on a regular grid.
        (CHECKED)
    
        Parameters:
        W (np.ndarray): Column vector of function values evaluated at each grid direction,
                        with the direction ordering given by grid2dirs. 
        aziRes (int): Azimuth resolution of the grid in degrees (same as used in grid2dirs function).
        elRes (int): Elevation resolution of the grid in degrees (same as used in grid2dirs function).
        CLOSED (int, optional): If true (1), the returned matrix replicates the first
                                column of function values at 0 degrees azimuth also at 360 degrees,
                                useful for 3D plotting so that the shape does not have a
                                hole in the end. Default is 0.
        
        Returns:
        np.ndarray: Matrix of the function values replicated on the grid points.
        """
        
        if 360 % aziRes != 0 or 180 % elRes != 0:
            raise ValueError('Azimuth or elevation resolution should divide exactly 360 and 180 degrees')
        
        Nphi = 360 // aziRes
        Ntheta = 180 // elRes + 1
        
        Nf = W.shape[1] if W.ndim > 1 else 1
        Wgrid = np.zeros((Nphi, Ntheta, Nf))
        
        W = np.expand_dims(W, axis=1)
        
        for i in range(Nf):       
            Wgrid[:, 1:-1, i] = np.reshape(W[1:-1, i], (Nphi, Ntheta-2), order='F')
            Wgrid[:, 0, i] = W[0]
            Wgrid[:, -1, i] = W[-1]
        
        if Nf != 1:
            Wgrid = np.transpose(Wgrid, (1, 0, 2))
        else:
            Wgrid = Wgrid.T
        
        if CLOSED:
            Wgrid = np.concatenate((Wgrid, Wgrid[:, :, 0][:, :, np.newaxis]), axis=2)
        
        return Wgrid if Nf != 1 else Wgrid.squeeze()
    
    
    @staticmethod
    def plotSphFunctionCoeffs(f_SH, order, aziRes=5, elRes=5, ax=None):
        
        """
        Draw a spherical function (real) defined on a grid.
        (CHECKED)
        
        Parameters:
        f_SH (numpy.ndarray): Vector of spherical harmonics coefficients of size (N+1)^2.
        order(int): Order of the spherical harmonics
        aziRes(int): Grid resolution in azimuth (degrees).
        elRes(int): Grid resolution in elevation (degrees).
        ax(matplotlib.axes): Axis handle for the plot; if not provided, a new one is created.
        """
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
        el, az = SH_visualization.grid2dirs(aziRes, elRes)
        
        # Function to visualize
        F = SHutils.inverse_SH(f_SH, order, el, az)
        # Corresponding function within a regular grid
        Fgrid = SH_visualization.Fdirs2grid(F, aziRes, elRes, 1)
    
        # Change to rad
        azi = np.deg2rad(np.arange(0, 360 + aziRes, aziRes))
        elev = np.deg2rad(np.arange(0, 180 + elRes, elRes)) # Change in case of elevation (not inclination)
        #elev = np.deg2rad(np.arange(-90, 90 + elRes, elRes))
        # Building the grid
        Az, El = np.meshgrid(azi, elev)
    
        
        Dx = np.cos(Az) * np.sin(El) * np.abs(np.squeeze(Fgrid))
        Dy = np.sin(Az) * np.sin(El) * np.abs(np.squeeze(Fgrid))
        Dz = np.cos(El) * np.abs(np.squeeze(Fgrid))
        
        # Real positive annd negative parts
        Dp_x = Dx * (Fgrid >= 0)
        Dp_y = Dy * (Fgrid >= 0)
        Dp_z = Dz * (Fgrid >= 0)
        Dn_x = Dx * (Fgrid < 0)
        Dn_y = Dy * (Fgrid < 0)
        Dn_z = Dz * (Fgrid < 0)
    
        # 3D axis
        maxF = np.max(np.abs(Fgrid))
        ax.plot([0, 1.1 * maxF], [0, 0], [0, 0], color='r')
        ax.plot([0, 0], [0, 1.1 * maxF], [0, 0], color='g')
        ax.plot([0, 0], [0, 0], [0, 1.1 * maxF], color='b')
    
        ax.plot_surface(Dp_x, Dp_y, Dp_z, color='b', alpha=0.7)
        ax.plot_surface(Dn_x, Dn_y, Dn_z, color='r', alpha=0.7)
    
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    
        ax.grid(True)
        plt.show()