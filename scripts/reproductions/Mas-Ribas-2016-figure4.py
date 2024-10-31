#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:58:59 2024

@author: john

This script reproduces Figure 4 (right) of Mas-Riba 2016.
"""

import sys
sys.path.append("../..")

import numpy as np

from dcbh_environment.cosmology import Cosmology


cosmology = Cosmology(z=3.1)

rs = 10**np.linspace(-3, 2, 100) # In units of Mpc / h
xi_dm = cosmology.dark_matter_correlation_function(rs)


#%% Plot results.
import matplotlib.pyplot as plt

plt.loglog(rs, xi_dm)
plt.xlim(0.001, 100)
plt.ylim(0.0001, 1000)
plt.grid()
plt.xlabel('r - Mpc h^-1')
plt.ylabel('xi')
plt.legend()