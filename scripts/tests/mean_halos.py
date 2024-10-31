#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:39:34 2024

@author: john
"""

import sys
sys.path.append("..")

import numpy as np
import time

from dcbh_environment.cosmology import Cosmology
from dcbh_environment.central_halo_environment import setup_bins
from dcbh_environment.central_halo_environment import CentralHaloEnvironment
from dcbh_environment.lyman_werner_radiation import LymanWernerRadiation08
from dcbh_environment.dark_matter_halos import HaloModel

# Set the cosmology parameters.
cosmology = Cosmology(z=10)
h = cosmology.Hubble

M     = 4e7 * h              # Central halo mass in solar masses / h
m_min = np.log10(4e7 * h)    # Minimum halo mass in log10 solar masses / h
m_max = np.log10(1e14 * h)   # Maximum halo mass in log10 solar masses / h
r_min = 0.002 * h            # Minimum separation in physical Mpc / h
r_max = 18  * h              # Maximum separation in physical Mpc / h

epsilon_dc = 0.1  # Duty cycle

N_mc = 20  # Number of Monte Carlo realisations
N_m  = 500  # Number of mass bins
N_r  = 100  # Number of radius bins


#%%
rng = np.random.default_rng()

rs, dr = setup_bins(r_min, r_max, N_r, scale='log10')
ms, dm = setup_bins(m_min, m_max, N_m, scale='log10')
logms, dlogm = setup_bins(m_min, m_max, N_m, scale='linear')

halo_model = HaloModel(cosmology, logms, dlogm, rs, dr, 'non-linear')

central_halo_env_params = (halo_model, M, ms, dm, rs, dr)
central_halo_env = CentralHaloEnvironment(*central_halo_env_params, rng)

#%%
import matplotlib.pyplot as plt

plt.imshow(np.log10(central_halo_env.mean_halos))