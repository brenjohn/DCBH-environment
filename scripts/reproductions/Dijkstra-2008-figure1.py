#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:34:16 2024

@author: john

This script reproduces figure 1 of Dijkstra 2008.
"""

import sys
sys.path.append("../..")

import numpy as np

from dcbh_environment.cosmology import Cosmology
from dcbh_environment.dark_matter_halos import HaloModel
from dcbh_environment.central_halo_environment import setup_bins

z = 10
cosmology = Cosmology(z=z)
h = cosmology.Hubble

M_min = np.log10(1.7e8 * h)  # Minimum halo mass in log10 solar masses / h
M_max = np.log10(1e14 * h)   # Maximum halo mass in log10 solar masses / h
r_min = h * 0.001 / (1+z)     # Minimum separation in physical Mpc / h
r_max = h * 100  / (1+z)     # Maximum separation in physical Mpc / h

ms, dm = setup_bins(M_min, M_max, 100)
rs, dr = setup_bins(r_min, r_max, 5000)


#%%
nonlinear_model = HaloModel(
    cosmology,
    ms,
    dm,
    rs,
    dr,
)


linear_model = HaloModel(
    cosmology,
    ms,
    dm,
    rs,
    dr,
    'linear'
)


#%% Plot results.
import matplotlib.pyplot as plt

rs = nonlinear_model.rs * (1+z) * h
correlation_values = [
    nonlinear_model.xi(10**M_min, 10**M_min, ri) for ri in range(len(rs))
]

correlation_values_linear = [
    linear_model.xi(10**M_min, 10**M_min, ri) for ri in range(len(rs))
]

plt.loglog(rs, correlation_values)
plt.loglog(rs, correlation_values_linear)
plt.xlim(0.01, 100)
# plt.ylim(0.0001, 1000)
plt.grid()
plt.xlabel('r - cm Mpc')
plt.ylabel('xi')
plt.legend()