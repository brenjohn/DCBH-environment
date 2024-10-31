#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 17:24:03 2024

@author: john
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

epsilon_dc = 0.1
num_halos = 100
num_samples = 100000

N = [(rng.random(num_halos) > epsilon_dc).sum() for i in range(num_samples)]

M = [rng.binomial(num_halos, 1-epsilon_dc) for i in range(num_samples)]

plt.hist(N, bins=50, density=True)
plt.hist(M, bins=50, density=True)