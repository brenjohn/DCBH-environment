#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 21:42:29 2024

@author: john
"""

import glob
import numpy as np
import matplotlib.pyplot as plt

sample_files = glob.glob('./samples_*.txt')

Js = [np.loadtxt(file) for file in sample_files]
Js = np.concatenate(Js)        

plt.hist(np.log10(Js), bins=50, density=True)
plt.yscale('log')
plt.xlabel('J_21,LW (log10)')
plt.ylabel('PDF')