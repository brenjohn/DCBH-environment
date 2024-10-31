#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:36:42 2024

@author: john
"""

import os
import sys
sys.path.append("../../../")

import numpy as np
import time

from dcbh_environment.cosmology import Cosmology
from dcbh_environment.central_halo_environment import setup_bins
from dcbh_environment.central_halo_environment import CentralHaloEnvironment
from dcbh_environment.lyman_werner_radiation import LymanWernerRadiation14
from dcbh_environment.dark_matter_halos import HaloModel

import multiprocessing as mp
from multiprocessing import Pool


def monte_carlo_simulation(environment_params, mc_params, seed, proc_id):
    """This function sets up and runs a Monte Carlo simulation for sampling
    the Lyman-Werner flux seen by a central halo. The simulation is summarised
    by eqn 5 of Dijkstra et al 2014.
    
    Samples are written in a txt file with the name "samples_{proc_id}.txt".
    """
    #========================== Monte Carlo Setup ============================#
    # Here we set up rng, HaloModel, CentralHalo_environment and 
    # LymanWernerRadiation objects to be used in the Monte Carlo simulation.
    # These are used for generating random numbers and looking up averages for
    # neighbouing halos and LW luminosities etc.
    rng = np.random.default_rng(seed)
    
    cosmology, M, m_min, m_max, N_m, r_min, r_max, N_r = environment_params
    rs, dr = setup_bins(r_min, r_max, N_r, scale='log10')
    ms, dm = setup_bins(m_min, m_max, N_m, scale='log10')
    logms, dlogm = setup_bins(m_min, m_max, N_m, scale='linear')
    
    halo_model = HaloModel(cosmology, logms, dlogm, rs, dr, 'non-linear')
    
    central_halo_env_params = (halo_model, M, ms, dm, rs, dr)
    central_halo_env = CentralHaloEnvironment(*central_halo_env_params, rng)
    central_halo_env.init_minimum_radius_bins()
    
    h = cosmology.Hubble
    z = cosmology.z
    halo_LW_intensity = LymanWernerRadiation14(cosmology, rng)
    halo_LW_intensity.init_mean_LW_intensities(ms / h, rs / h)
    
    # Here we unpack some Monte Carlo related parameters and open a file to
    # write results to.
    N_mc, epsilon = mc_params
    results = open(f'./z_{z}/samples_{proc_id}.txt', 'a')
    
    
    #======================== Monte Carlo Simulation =========================#
    
    ti = time.time()
    # Generate N_mc Monte Carlo samples.
    for sample in range(N_mc):
        J_LW = 0
        
        # Loop over mass/radius bins
        for mi in range(N_m):
            r_min = central_halo_env.minimum_radius_bin[mi]
            for rj in range(r_min, N_r):
                # Get number of halos in this bin
                num_halos = central_halo_env.num_halos(mi, rj)
                
                # Compute the LW flux from the star forming halos in this bin.
                if num_halos > 0:
                    if num_halos < 10:
                        star_forming_halos = rng.binomial(num_halos, epsilon)
                        J_LWs = halo_LW_intensity(mi, rj, star_forming_halos)
                        J_LW += J_LWs.sum()
                    else:
                        mean_J_LW = halo_LW_intensity[mi, rj]
                        J_LW += num_halos * epsilon * mean_J_LW
        
        # Convert J_LW to 1e−21 erg / (s cm^2 sr Hz) and write result to disk.
        J_LW *= 1e21
        results.write(str(J_LW) + '\n')
    
    tf = time.time()
    results.close()
    with open(f'./proc_{proc_id}_stats.txt', 'a') as sim_stats:    
        sim_stats.write(f'{N_mc} samples generated in {tf-ti} seconds\n')



if __name__ == "__main__":
    
    # Set the number of processors and the rng seed.
    num_procs, seed = 6, int(time.time())
    
    # Set the cosmology parameters.
    cosmology = Cosmology(
        z         = 20,
        Om0       = 0.3175,
        Ode0      = 0.6825,
        Ob0       = 0.04899,
        H0        = 67.11,
        sigma_8   = 0.8344,
        ns        = 0.9624,
    )
    h = cosmology.Hubble

    M     = 4e7 * h              # Central halo mass in solar masses / h
    m_min = 5 * h                # Minimum halo mass in log10 solar masses / h
    m_max = 15 * h               # Maximum halo mass in log10 solar masses / h
    r_min = np.log10(0.002 * h)  # Minimum separation in log10 Mpc / h
    r_max = np.log10(66  * h)    # Maximum separation in log10 Mpc / h

    epsilon_dc = 0.2  # Duty cycle

    N_mc = 1000  # Number of Monte Carlo realisations
    N_m  = 400  # Number of mass bins
    N_r  = 100  # Number of radius bins
    
    # Pack the parameters for the underlying halo model into a tuple to be
    # passed to worker processes.
    environment_params = (
        cosmology,
        M,
        m_min,
        m_max,
        N_m,
        r_min,
        r_max,
        N_r
    )
    
    # Pack the Monte Carlo parameters into a tuple for the worker processes.
    mc_params = N_mc, epsilon_dc
    
    # Create the parameter tuples to be passed to worker processes.
    work = [
        (environment_params, mc_params, seed + proc, proc) 
        for proc in range(num_procs)
    ]
    os.makedirs(f'./z_{cosmology.z}/', exist_ok=True)
    
    # Create a pool of worker processes and start them working on MC sims.
    with mp.get_context('spawn').Pool(num_procs) as pool:
        pool.starmap(monte_carlo_simulation, work)
    # monte_carlo_simulation(*work[0]) # for debugging
