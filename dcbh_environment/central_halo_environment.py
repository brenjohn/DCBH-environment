#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:18:50 2024

@author: john
"""

import numpy as np


class CentralHaloEnvironment():
    """The CentralHaloEnvironment class has attributes and methods related to
    mass and radius bins used in Monte Carlo simulations of the kind described
    in Dijkstra 2008 and 2016.
    """
    
    def __init__(self, halo_model, M, ms, dm, rs, dr, rng=None):
        self.halo_model = halo_model
        self.M  = M   # central halo mass.
        self.ms = ms  # Mass bin interior points.
        self.dm = dm  # Mass bin widths.
        self.rs = rs  # Radius bin interior points.
        self.dr = dr  # Radius bin widths.
        
        self.cosmology = halo_model.cosmology
        self.rng = rng if rng else np.random.default_rng()
        self.initialise_mean_halos()
        self.initialise_shell_areas()
        self.init_metal_pollution_parameters()
        
    #=========================================================================#
    #                       Mass-Radius Bin Methods
    #=========================================================================#
        
    def initialise_mean_halos(self):
        """Initialises an array containing the mean number of halos in each
        mass-radius bin. This method relies on eqn 1 from Dijkstra 2008.
        """
        dndm = self.halo_model.dndm
        M    = self.M
        dr   = self.dr
        rs   = self.rs
        ms   = self.ms
        dm   = self.dm
        
        Nm, Nr = len(ms), len(rs)
        mean_halos = np.zeros((Nm, Nr))
        
        for mi, m in enumerate(ms):
            for rj in range(Nr):
                shell_volume = 4 * np.pi * rs[rj]**2 * dr[rj]
                num_density  = dndm[mi] * dm[mi]
                num_density *= 1 + self.halo_model.xi(M, m, rj)
                mean_halos[mi, rj] = num_density * shell_volume
                
        self.mean_halos = mean_halos
        
    
    def initialise_shell_areas(self):
        """Initialises an array containing the surface area of a sphere whose
        radius equals the radius of each radius bin. The area is calculated in
        cm**2. This can be used to convert luminosities nto intensities.
        """
        self.sphere_areas = [
            4 * np.pi * r**2 
            for r in self.cosmology.mpc_in_cm * self.rs / self.cosmology.Hubble
        ]
    
    
    def num_halos(self, mi, rj):
        """Returns the number of halos of a given mass-radius bin as a sample
        from a poisson distribution. 
        
        The mean number of halos is taken from the look-up table created in the
        "initialise_mean_halos" method.
        """
        mean_halos = self.mean_halos[mi, rj]
        num_halos = self.rng.poisson(mean_halos)
        return num_halos
    
    
    #=========================================================================#
    #                         Metal Pollution Methods
    #=========================================================================#
    
    def init_metal_pollution_parameters(self):
        # Paramerers related to metal pollution radius
        self.nu     = 0.01   # Number of supernovae formed per mass (1/Msolar)
        self.E_0    = 1e51   # Supernova explosion energy (ergs = g*cm^2/s^2)
        self.Delta  = 60     # overdensity parameter
        self.f_star = 0.05   # fraction of baryonic mass that form stars

        # (m_p)(n): density of the gas, units: g/(cm^3)
        # Critical density at z=0 and Density of the gas at z
        O_b, z = self.cosmology.O_b, self.cosmology.z
        self.rho_crit = self.cosmology.cosmo_model.critical_density(z=0).value
        self.rho_gas  = self.Delta * O_b * self.rho_crit * (1 + z)**3
        
        self.wind_factor = 1
        
    
    def metal_enriched_radius(self, M, t):
        """
        Radius of metal enriched region produced by supernova in a neighbouring
        dark matter halo. This is computed using eqn 4 in Dijkstra 2014.
        
        parameters:
            M - mass of halo which hosts the supernova in solar masses
            t - time since supernova in yrs
        returns:
            radius of metal enriched region in Mpc
        """
        # Units: M_star = M_solar, rs = cm
        Ob_over_Om = self.cosmology.O_b / self.cosmology.O_m
        M_star = self.f_star * (Ob_over_Om) * M
        rs = (self.E_0 * self.nu * M_star / self.rho_gas)**(1/5) * t**(2/5)
        return rs * self.cosmology.cm_in_mpc
    
    
    def minimum_separation(self, M):
        """Returns the minimum separation between halos while accounting for 
        metal pollution from galatic outflows. See section 2 and 3.3 of Dijktra
        2014.
        
        parameters:
            M - mass of target halo in solar masses
        returns:
            minimum separation in Mpc
        """
        # Compute the minimum possible separation between halos of mass M.
        h = self.cosmology.Hubble
        r_vir = 1e3 * self.halo_model.virial_radius(M) * 1e-6 / h
        r_min = 2 * r_vir
        
        # Compute the radius of metal polluted region (Mpc).
        t_ff = self.cosmology.freefall_time() * 1e6
        r_s = self.metal_enriched_radius(M, t_ff)
        
        # Including/excluding metals:
        r_min = max(r_min, self.wind_factor * r_s)
        
        return r_min
    
    
    def init_minimum_radius_bins(self):
        """Creates an array of radius bin indices corresponding to the radius
        bin of smallest radius allowed by metal pollution for each mass bin.
        """
        self.minimum_radius_bin = np.zeros_like(self.ms, dtype=int)
        
        h = self.cosmology.Hubble
        for mi, m in enumerate(self.ms):
            r_min = self.minimum_separation(m / h)
            r_min_i = np.where(self.rs > r_min * h)[0][0]
            self.minimum_radius_bin[mi] = int(r_min_i)
    


def setup_bins(start, end, num_bins, scale='linear'):
    """
    """
    # Get bin centres and edges.
    bin_edges = np.linspace(start, end, num_bins + 1)
    centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # If using a nonlinear scale, convert to linear scale.
    if scale == 'log10':
        bin_edges = 10**bin_edges
        centres = 10**centres
        
    widths = bin_edges[1:] - bin_edges[:-1]
    return centres, widths