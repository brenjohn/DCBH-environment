#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:18:44 2024

@author: john
"""

import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

import numpy as np
import matplotlib.pyplot as plt

from dcbh_environment.cosmology import Cosmology
from dcbh_environment.central_halo_environment import setup_bins
from dcbh_environment.dark_matter_halos import HaloModel

from scipy.interpolate import InterpolatedUnivariateSpline as spline


#%%
Om0     = 0.3175
Ode0    = 0.6825
Ob0     = 0.04899
H0      = 67.11
sigma_8 = 0.8344
ns      = 0.9624
h       = H0 / 100

m_min = np.log10(4e7 / h)
m_max = np.log10(1e15 / h)
N_m = 500
logms, dlogm = setup_bins(m_min, m_max, N_m, scale='linear')

r_min = 0.002
r_max = 66
N_r = 100
rs, dr = setup_bins(r_min, r_max, N_r, scale='linear')

zs = np.arange(10, 21, 2)

n_DCBH_30, n_DCBH_100, n_DCBH_300 = [], [], []


#%%
P_pristine = None

with open("TS2009_Fig1_LR__26_04_2023.csv") as file:
    # Data extrapolated from Trenti and Stiavelli (2009),
    # Figure 1. (lower right).
    TS09_data = np.loadtxt(file, delimiter=",")
    TS09_zs = TS09_data[:, 0]
    TS09_Ps = TS09_data[:, 1]
    
    # Defining P_pristine as a function:
    P_seed_spl = spline(TS09_zs, TS09_Ps)
    P_pristine = lambda z: 1 - P_seed_spl(z)


#%%

for z in zs:
    print(f'Computing n_DCBH for z = {z}')
    cs = Cosmology(
        z       = z,
        Om0     = Om0,
        Ode0    = Ode0,
        Ob0     = Ob0,
        H0      = H0,
        sigma_8 = sigma_8,
        ns      = ns
    )
    hm = HaloModel(cs, logms, dlogm, rs, dr)
    
    # Number density of halos with mass above m_min in 1/cMpc**3
    n = h**3 * np.trapz(hm.dndm, 10**hm.logms) * (1 + z)**3
    
    P_gen = P_pristine(z)
    
    n_DCBH = n * P_gen
    n_DCBH_30.append(n_DCBH)
    
    n_DCBH = n * P_gen
    n_DCBH_100.append(n_DCBH)
    
    n_DCBH = n * P_gen
    n_DCBH_300.append(n_DCBH)
    

#%%
plt.plot(zs, n_DCBH_30)
plt.plot(zs, n_DCBH_100)
plt.plot(zs, n_DCBH_300)
plt.yscale('log')