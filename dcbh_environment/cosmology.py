#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:21:40 2023

@author: john

This file defines a cosmology class for:
    a) storing and computing relevant cosmology related parameters,
    b) creating Transfer function and DMHaloModel objects from the stored
       parameters.
"""
from numpy import pi, sin

from astropy import units
from astropy.cosmology import Planck13
from astropy.cosmology import WMAP9
from astropy.cosmology import FlatLambdaCDM

from scipy.integrate import trapezoid

from hmf.density_field.transfer import Transfer
from hmf.mass_function.fitting_functions import SMT
from hmf.mass_function.fitting_functions import PS

from halomod.bias import SMT01
from halomod.bias import ST99
from halomod.halo_model import DMHaloModel


class Cosmology():
    
    def __init__(
            self, 
            z       = 10,
            Om0     = 0.27,     # Matter density parameter
            Ode0    = 0.73,     # Dark energy density parameter
            Ob0     = 0.042,    # Baryonic matter density parameter
            H0      = 70,       # Hubble constant h in km/s/Mpc
            sigma_8 = 0.8344, 
            ns      = 0.9624,
            Tcmb0   = 2.725
        ):
        # self.cosmo_model = Planck13.clone(
        #     name="Planck13 from D14", 
        #     H0    = 67.11, 
        #     Om0   = 0.3175, 
        #     Tcmb0 = 2.725, 
        #     Neff  = 3.05,
        #     m_nu  = [0., 0., 0.] * units.eV, 
        #     Ob0   = (0.022068)/(0.6711**2)
        # )
        # self.cosmo_model = Planck13.clone(
        #     name="Planck13 from D14", 
        #     H0    = 70.00, 
        #     Om0   = 0.27,
        #     Ob0   = 0.042,
        # )
        
        # Create the custom FlatLambdaCDM cosmology
        self.cosmo_model = FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0, Tcmb0=Tcmb0)
        self.cosmo_model.sigma8 = sigma_8  # Add sigma8 as an attribute
        
        self.z       = z                      # Redshift
        self.sigma_8 = sigma_8
        self.n       = ns
        self.Hubble  = self.cosmo_model.h     # Dimensionless Hubble constant
        self.O_l     = self.cosmo_model.Ode0  # Dark energy density
        self.O_m     = self.cosmo_model.Om0   # Matter density
        self.O_b     = self.cosmo_model.Ob0   # Baryonic matter density
        
        self.cm_in_mpc = units.cm.to('Mpc')
        self.mpc_in_cm = units.Mpc.to('cm')
        
        
    def transfer_function(self, dlnk=0.0001):
        """Creates a hmf Transfer object from the stored and given parameters.
        """
        return Transfer(
            cosmo_model = self.cosmo_model, 
            sigma_8     = self.sigma_8, 
            z           = self.z, 
            n           = self.n,
            dlnk        = dlnk
        )
    
    
    def dark_matter_halo_model(self, M_min, M_max, dM):
        """Creates a halomod DMHaloModel object from the stored and given
        parameters.
        """
        return DMHaloModel(
            cosmo_model = self.cosmo_model, 
            sigma_8     = self.sigma_8, 
            hmf_model   = SMT,
            bias_model  = SMT01,
            z           = self.z, 
            Mmin        = M_min, 
            Mmax        = M_max,
            dlog10m     = dM,
            n           = self.n
        )
    
    
    def dark_matter_correlation_function(self, rs, spectrum='non-linear'):
        """Return the dark-matter correlation function following Eqn 21 in 
        Mas-Riba 2016. (eqn B2 in the arxiv version)
        
        This correlation function is computed by integrating the power spectrum
        P(k) over momentum k. It comes from an inverse 3D Fourier 
        transform. We assume that P(k) is spherically symmetric so this 
        3D integral is reduced to be over 1 variable only.
        
        parameters:
            cosmology - A Cosmology object.
            rs        - seperations in Mpc / h to get the correlations for.
            spectrum  - A string determining the type of power spectrum to use. 
        """
        tfr = self.transfer_function()
        Ps = tfr.power if spectrum == "linear" else tfr.nonlinear_power
        ks = tfr.k # in units of h/Mpc
        xi_dm_factor = (ks**2) * Ps / (ks * 4 * pi**2)
        
        def xi_dm(r):
            xi_dm_integrand = xi_dm_factor * sin(ks * r) / r
            return trapezoid(xi_dm_integrand, ks)
        
        return [xi_dm(r) for r in rs]
 
    
    def freefall_time(self):
        """The free-fall time in Myr.
        
        See 3rd paragraph of appendix B of Dijkstra 2014.
        """
        return 83 * ((1 + self.z) / 11)**(-3/2)
    
    
    def freefall_time_seconds(self):
        """The free-fall time in seconds.
        """
        t_ff = self.freefall_time()
        return (t_ff * (1e6) * units.yr).to('s').value