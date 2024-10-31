#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 20:16:03 2023

@author: john

This module models the Lyman-Werner luminosity of galaxies as outlined in
section 3.1 in Dijkstra 2014.
"""

import numpy as np
import astropy
import astropy.units as units

from scipy import integrate
from .cosmology_model import Cosmology


class LymanWernerModel():
    
    def __init__(self, cosmology):
        
        self.cosmology = cosmology if cosmology else Cosmology()
        
        self.z      = cosmology.z      # Redshift
        self.Hubble = cosmology.Hubble # Dimensionless Hubble constant
        self.O_m    = cosmology.O_m    # Matter density parameter
        self.O_b    = cosmology.O_b    # Baryonic matter density parameter
        self.t_ff   = self.cosmology.freefall_time()
        
        # TODO: Hannah used the hubble constant for h below but it seems from 
        # Ahn et al it should be Plank's constant 
        # The below constant is used for the picket fence modulation factor 
        # (see eqn 23 of Ahn 2009).
        h = astropy.constants.h.value
        self.alpha = (0.7 / h) * np.sqrt((0.27 * 21) / (self.O_m * (1+self.z)))
        
        # Compute the mean energy of a LW photon per Hz in the LW band
        h      = astropy.constants.h
        mean_E = 12.4 * units.eV
        del_E  = (13.6 - 11.2) * units.eV
        self.energy_per_hz = (h * mean_E / del_E).to('erg / Hz').value
        
        # paramters for halo UV luminosity. See section 2.1.2 Dijkstra 2008
        self.sigma_LW = 0.4
        self.F        = 1.0


    def mean_LW_photon_production_rate(self, t):
        """
        We use the following formula to compute the mean LW photon production
        rate per solar mass of stars formed at time t. See section 3.1 of 
        Dijkstra 2014.
        
        parameters:
            t - time in yrs
        Returns:
            LW photon production rate in (1/s) * (1/M_solar)
        """
        Q_0 = (1e47) # / self.Hubble # TODO: Do I need Hubble here?
        t6 = t / (10**6)
        return Q_0 * (1 + t6 / 4)**(-3/2) * np.exp(-t6 / 300)


    def mean_LW_luminosity_density(self, M, t):
        """
        This function uses eqn 8 from section 3.1 of Dijkstra 2014 to compute 
        the mean LW luminosity density for a given halo of mass M and at time 
        t.
        
        In this model, we assume a fraction eplison_DC of all dark matter halos
        of mass M have galaxies, of gaseous mass Mg = (Omega_b/Omega_m) * M, 
        generate a LW background through a burst of young star formation. We 
        assume a fraction f_star of the gaseous mass is converted into stars 
        (giving stellar mass f_star * Mg). We set f_star = 0.05 (see paper).
        
        parameters:
            M - mass in solar masses
            t - time in yrs 
        Returns:
            mean LW luminosity in (erg / Hz) * (1 / s)
        """
        # Compute the mass of stars in the given halo of mass M.
        Mg = (self.O_b / self.O_m) * M
        f_star = 0.05
        M_star = f_star * Mg
        
        # Estimate the number of LW photons, per solar mass of stars formed, 
        # and the fraction of them that escape the galxy.
        Q_LW  = self.mean_LW_photon_production_rate(t)
        f_esc = 1
        
        # Note the units of the following should be: 
        #          (  erg / Hz   ) * (         1 / s       )
        return self.energy_per_hz  *  Q_LW * f_esc * M_star
    
    
    def mean_UV_luminosity(self, M):
        """
        Eqn 2 from Dijkstra 2008
        
        parameters:
            M - mass in solar masses
        returns:
            UV luminosity in 10**26 * erg 1/(s * Hz)
        """
        return 2.8 * (M/1e8) * (11/(1+self.z))**(2/3)


    def pdf_halo_UV_luminosity(self, L, M):
        """
        Returns the probability that a halo of mass M has UV luminosity in the
        range log(L) + d(log(L)).
        
        The spread in UV luminosity produced in halos of mass M is modeled with 
        a log–normal distribution P(L, M). See section 2.1.2 of Dijkstra 2008.
        
        To compute the probability density, the mean LW luminosity is computed
        at the free fall time (see section 3.1 of Dijkstra 2014.). Also, 
        following Dijkstra 2014, sigma_LW is set to 0.4.
        
        parameters:
            L - luminosity in 10^26 erg / (s * Hz)
            M - mass in solar masses
        """
        # Compute the mean LW luminosity in the correct units.
        mean_LW = self.mean_LW_luminosity_density(M, self.t_ff * 1e6) / 1e26
        # mean_LW = self.mean_UV_luminosity(M)
        
        # Evaluate the probability density (eqn 3 in Dijkstra 2008).
        exponent = np.log10(L / (self.F * mean_LW))**2 / (2 * self.sigma_LW**2)
        return np.exp(-exponent) / (self.sigma_LW * np.sqrt(2 * np.pi) )
    
    
    def differential_pdf_halo_UV_luminosity(self, L, M):
        """
        Derivitive of pdf_halo_UV_luminosity with respect to log10(L)
        
        parameters:
            L - luminosity in 10^26 erg / (s * Hz)
            M - mass in solar masses
        """
        mean_LW = self.mean_LW_luminosity_density(M, self.t_ff * 1e6) / 1e26
        
        log_diff = np.log10(L / (self.F * mean_LW))
        divisor = self.sigma_LW**3 * np.sqrt(2 * np.pi)
        
        exponent = -log_diff**2 / (2 * self.sigma_LW**2)
        return -log_diff * np.exp(exponent) / divisor


    def intensity_to_luminosity(self, J, r):
        """
        Convert the intensity J to a luminosity L for a spacial seperation r.
        
        The intensity J has units (erg / s) /  (cm**2 * Hz * sr). It's the flux
        recieved per unit area per Hz per solid angle. Note the solid angle 
        comes from the fact that for each point in the unit area which recieves
        flux, flux comes from all points in the distant galaxy which subtends 
        some solid angle. In otherwords, it's a consequence of the galaxy not
        being a point source.
        
        To convert the average intensity into a luminosity (ie the total flux 
        radiated by the source), we multiply by the solid angle 4*pi and the
        surface area of a sphere of radius r. We then divide by f_mod which
        accounts for extra dimming introduced by the LW-horizon (see Ahn 2009 
        for further details)
        
        See eqn 7 in section 3.1 of Dijkstra 2014.
        
        parameters:
            J - intensity (erg / s) /  (cm**2 * Hz * sr)
            r - spacial separation - Mpc
        returns:
            luminosity in erg /  (s * Hz)
        """
        # Damping factor f_mod accounts for LW photons being redshifted from 
        # source to target halo
        # TODO: Hannah had an extra factor of 1/Hubble for R, check this with 
        # her.
        R =  (1 + self.z) * r # Convert to cMpc
        f_mod = self.picket_fence_modulation_factor(R)
        
        # convert radius to cm
        r_cm = (r * units.Mpc).to('cm').value
        # units   (sr)     *   (cm**2)  * (erg / s) / (cm**2 * Hz * sr)
        return (4 * np.pi) * (4 * np.pi * r_cm**2) * (J / f_mod)
    
    
    def luminosity_to_intensity(self, L, r):
        """
        The inverse of intensity_to_luminosity
        TODO: Add a test which checks if these are in fact inverses.
        
        Parameters:
            L - luminosity in (erg / s) * (1/Hz)
            r - radius in Mpc
        Returns:
            intensity J in (erg / s) /  (cm**2 * Hz * sr)
        """
        R =  r * (1 + self.z) # Convert to cMpc
        f_mod = self.picket_fence_modulation_factor(R)
        r_cm = (r * units.Mpc).to('cm').value
        return L * f_mod / (16 * np.pi**2 * r_cm**2)
        


    def picket_fence_modulation_factor(self, r):
        """Fitting function for picket fence modulation factor (Ahn et al 2009).
        
        parameters:
            r - distance in cMpc
        """
        if r > 97.39 * self.alpha:
            return 0
        return 1.7 * np.exp(-(r / (116.29 * self.alpha))**0.68) - 0.7
    
    
    
    def prob_flux_greater_than_crit(self, M, L_crits):
        """
        parameters:
            M - mass in solar mass.
            L_crits - critical flux values in 10^26 (erg / s) * (1/Hz)
        """
        Ls = 10**np.linspace(-4, 3, 100)
        Ls = np.concatenate((Ls, L_crits))
        Ls = np.unique(Ls)
        Ls = np.sort(Ls)
        crit_inds = np.where(np.isin(Ls, L_crits))[0]
        
        log_Ls = np.log10(Ls)
        dpdlogL = [self.pdf_halo_UV_luminosity(L, M) for L in Ls]
        cdf = integrate.cumulative_trapezoid(dpdlogL, log_Ls, initial = 0)
        
        return cdf[-1] - cdf[crit_inds]
    