#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:09:38 2024

@author: john

This file defines the HaloModel class which wraps the DMHaloModel class from
the halomod package. Note, we use composition rather than inheritiance to wrap
the DMHaloModel class.
"""
# TODO: the halo mod and hmf packages probably give the correlation functions
# We need and could save computing them for other quantities.
import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline as spline


class HaloModel():
    """The HaloModel class wraps the DMHaloModel class from the halomod
    package. 
    
    It adds functionality for computing the halo-halo correlation
    function described in Dijkstra et all 2008 section 2.1.1. Note, this
    correlation function is only computed for the mass and radius ranges
    """
    
    def __init__(
            self, 
            cosmology, 
            logms, 
            dlogm,
            rs,
            dr,
            spectrum = 'non-linear'
        ):
        """
        Mass paramters should have units log10( solar mass / h).
        Radius/separation parameters should have units Mpc / h.
        """
        self.cosmology = cosmology
        self.logms = logms
        self.dlogm = dlogm
        self.rs = rs
        self.dr = dr
        
        self.xi_dm = cosmology.dark_matter_correlation_function( 
            self.rs, spectrum
        )
        
        self.halo_model = cosmology.dark_matter_halo_model(
            self.logms[0], self.logms[-1], self.dlogm[0]
        )
        
        self.dndm = self.halo_model.dndm
        self.D_plus = self.compute_linear_growth_factor()
        self.init_bias()
        
        
    def init_bias(self):
        self.bias_array = self.halo_model.halo_bias
        
        ms = self.halo_model.m
        self.bias = spline(ms, self.bias_array)
        
    
    def compute_linear_growth_factor(self):
        """Returns the linear growth factor
        
        TODO: Ask Hannah where this formula comes from and then reference it 
        here. this was pulled from Hannah's correlations file. There's mention 
        of the factor in eqn 17 of D14
        """
        prefactor    = lambda x: ((1 + x**3)**0.5) / (x**1.5)
        integrand    = lambda x: (x**1.5) / ((1 + x**3)**1.5)
        scale_factor = lambda z: 1 / (1 + z)
        
        O_l = self.cosmology.O_l
        O_m = self.cosmology.O_m
        z   = self.cosmology.z
    
        x_z   = (O_l / O_m)**(1/3) * scale_factor(z)
        xs_z  = np.linspace(0.0, x_z, 10000)
    
        x_10  = (O_l / O_m)**(1/3) * scale_factor(10)
        xs_10 = np.linspace(0.0, x_10, 10000)
    
        numerator   = prefactor(x_z)  * np.trapz(integrand(xs_z),  xs_z)
        denominator = prefactor(x_10) * np.trapz(integrand(xs_10), xs_10)
    
        return numerator / denominator
    
    
    def xi(self, M, m, rk):
        """Returns the correlation function used in Dijkstra 2008 and 2016.
        
        mass parameters should be in solar masses / h.
        """
        return self.bias(M) * self.bias(m) * self.xi_dm[rk] * self.D_plus**2
    
    
    def virial_radius(self, M):
        """Returns the virial radius according to eqn 12 in Dijkstra 2014.
        
        parameters:
            M - mass of target halo in solar masses
        returns:
            virial radius in kpc / h
        """
        M = M * self.cosmology.Hubble / 1e8 # Convert to 10**8 solar mass / h
        O_m, z = self.cosmology.O_m, self.cosmology.z
        return 0.784 * O_m**(-1/3) * 10/(1 + z) * M**(1/3)