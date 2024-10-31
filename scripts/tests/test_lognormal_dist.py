#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:07:49 2024

@author: john
"""

import numpy as np
import matplotlib.pyplot as plt


num_samples = 100

mu = 10
sigma = 2
xs = 10**(mu + sigma * np.random.randn(num_samples))

mu_prime = np.log(10) * mu
sigma_prime = np.log(10)*sigma
ys = np.random.lognormal(mu_prime, sigma_prime, num_samples)

plt.hist(np.log10(xs), bins=50)
# plt.show()
# plt.close()

plt.hist(np.log10(ys), bins=50)
# plt.show()
# plt.close()

rng = np.random.default_rng()
