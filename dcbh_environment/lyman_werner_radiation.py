#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 21:19:02 2024

@author: john
"""

import numpy as np

import astropy
from astropy import units


class LymanWernerRadiation08():
    """This class can be used to model the Lyman-Werner radiation from star
    forming halos as outlined in the Dijkstra 2008 paper.
    """
    
    def __init__(self, cosmology, rng=None):
        self.cosmology = cosmology
        self.rng = rng if rng else np.random.default_rng()
        
        self.init_D08_attributes()
        
    
    def init_D08_attributes(self):
        """Initialises variables described in Dijkstra 2008
        """
        # paramters for halo UV luminosity. See section 2.1.2 Dijkstra 2008
        self.sigma_LW = 0.5 * np.log(10)
        self.F        = 0.38
    
    
    def mean_UV_luminosity(self, M):
        """computes the mean Lyman-Werner luminosity of a star forming halo
        of mass M. This function relys on eqn 2 from Dijkstra 2008.
        
        Note: 
            - The given M should have units of solar masses
            - The returned UV luminosity are in 10**26 * erg 1/(s * Hz)
        """
        return 2.8 * (M/1e8) * (11/(1+self.cosmology.z))**(3/2)
    
    
    def init_luminosities(self, masses):
        """Initialises two arrays of mean luminosities for the given masses.
        The given masses are taken to represent mass bins.
        """
        self.mean_luminosities = [
            self.F * self.mean_UV_luminosity(m) for m in masses
        ]
        # Note, the factor of log(10) is to account for the change of base when
        # using the rng.lognormal method.
        self.log_mean_luminosities = [
            np.log(10) * np.log10(l) for l in self.mean_luminosities
        ]
    
    
    def __call__(self, mi, num_samples):
        """Returns samples from a lognormal distribution modelling the
        Lyman-Werner luminosity of halos in the given mass bin.
        """
        return self.rng.lognormal(
            self.log_mean_luminosities[mi], self.sigma_LW, num_samples
        )
    
    
    def __getitem__(self, mi):
        """Return the mean Lyman-Werner luminosity for the given mass bin.
        """
        return self.mean_luminosities[mi]
    
    

class LymanWernerRadiation14():
    """This class can be used to model the Lyman-Werner radiation from star
    forming halos as outlined in the Dijkstra 2014 paper.
    """
    
    def __init__(self, cosmology, rng=None):
        self.cosmology = cosmology
        self.rng = rng if rng else np.random.default_rng()
        
        self.init_D14_attributes()
    
    
    def init_D14_attributes(self):
        """Initialises variables described in Dijkstra 2014.
        """
        self.sigma_LW = 0.4 * np.log(10)
        
        h = astropy.constants.h.value
        O_m = self.cosmology.O_m
        z = self.cosmology.z
        self.alpha = (0.7 / h) * np.sqrt((0.27 * 21) / (O_m * (1 + z)))
        
        # Compute the mean energy of a LW photon per Hz in the LW band
        h      = astropy.constants.h
        mean_E = 12.4 * units.eV
        del_E  = (13.6 - 11.2) * units.eV
        self.energy_per_hz = (h * mean_E / del_E).to('erg / Hz').value
    
    
    def mean_LW_photon_production_rate(self, t):
        """We use the following formula to compute the mean LW photon 
        production rate per solar mass of stars formed at time t. See section 
        3.1 of Dijkstra 2014.
        
        parameters:
            t - time in yrs
        Returns:
            LW photon production rate in (1/s) * (1/M_solar)
        """
        Q_0 = (1e47) # (solar mass)**-1 s**-1
        t6 = t / (10**6)
        return Q_0 * (1 + t6 / 4)**(-3/2) * np.exp(-t6 / 300)


    def mean_LW_luminosity_density(self, M, t):
        """This function uses eqn 8 from section 3.1 of Dijkstra 2014 to 
        compute the mean LW luminosity density for a given halo of mass M and 
        at time t.
        
        parameters:
            M - mass in solar masses
            t - time in yrs 
        Returns:
            mean LW luminosity in (erg / Hz) * (1 / s)
        """
        # Compute the mass of stars in the given halo of mass M.
        Mg = (self.cosmology.O_b / self.cosmology.O_m) * M
        f_star = 0.05
        M_star = f_star * Mg
        
        # Estimate the number of LW photons, per solar mass of stars formed, 
        # and the fraction of them that escape the galaxy.
        Q_LW  = self.mean_LW_photon_production_rate(t)
        f_esc = 1
        
        # Note the units of the following should be: 
        #          (  erg / Hz   ) * (         1 / s       )
        return self.energy_per_hz  *  Q_LW * f_esc * M_star
    
    
    def luminosity_to_intensity(self, L, r):
        """Converts luminosity to intensity.
        
        Parameters:
            L - luminosity in (erg / s) * (1/Hz)
            r - radius in Mpc
        Returns:
            intensity J in (erg / s) /  (cm**2 * Hz * sr)
        """
        R =  r * (1 + self.cosmology.z) # Convert to cMpc
        f_mod = self.picket_fence_modulation_factor(R)
        r_cm = r * self.cosmology.mpc_in_cm
        return L * f_mod / (16 * np.pi**2 * r_cm**2)
    
    
    def picket_fence_modulation_factor(self, r):
        """Fitting function for picket fence modulation (Ahn et al 2009).
        
        parameters:
            r - distance in cMpc
        """
        if r > 97.39 * self.alpha:
            return 0
        return 1.7 * np.exp(-(r / (116.29 * self.alpha))**0.68) - 0.7
    
    
    def init_mean_LW_intensities(self, masses, radii):
        """Returns an array containing the mean Lyman-Werner intensities using
        eqn 7 from Dijkstra 2014. A mean intensity is computed for each given
        mass-radius bin.
        
        Note: Intensities are in (erg / s) /  (cm**2 * Hz * sr)
        Parameters:
            masses - mass of mass bins in solar masses
            radii  - radius of radius bins in Mpc
        Returns:
            mean intensity in (erg / s) /  (cm**2 * Hz * sr)
        """
        Nm, Nr = len(masses), len(radii)
        mean_LW_intensity = np.zeros((Nm, Nr))
        
        # Free fall time in yrs.
        tff = self.cosmology.freefall_time() * 1e6
        
        for mi, m in enumerate(masses):
            L = self.mean_LW_luminosity_density(m, tff)
            
            for rj, r in enumerate(radii):
                J = self.luminosity_to_intensity(L, r)
                mean_LW_intensity[mi, rj] = J
                
        self.mean_LW_intensity = mean_LW_intensity
        
        
    def __call__(self, mi, rj, num_samples):
        """Returns samples from a lognormal distribution modelling the
        Lyman-Werner intensity of halos in the given mass-radius bin.
        """
        mean_intensity = self.mean_LW_intensity[mi, rj]
        return mean_intensity * self.rng.lognormal(
            0, self.sigma_LW, num_samples
        )
    
    
    def __getitem__(self, index):
        """Return the mean Lyman-Werner intensity for the given mass-radius 
        bin.
        """
        mi, rj = index
        return self.mean_LW_intensity[mi, rj]