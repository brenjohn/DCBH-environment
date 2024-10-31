#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:05:15 2024

@author: john

This script generates samples of Lyman Werner flux seen by a candidate halo for
a direct collapse black hole. The Monte Carlo method used for generating the
samples is outlined in Dijkstra et al 2008 (Fluctuations in the High–Redshift 
Lyman–Werner Background: Close Halo Pairs as the Origin of Supermassive Black 
Holes).
"""

import sys
sys.path.append("../../../")

import numpy as np
import time

from dcbh_environment.cosmology import Cosmology
from dcbh_environment.central_halo_environment import setup_bins
from dcbh_environment.central_halo_environment import CentralHaloEnvironment
from dcbh_environment.lyman_werner_radiation import LymanWernerRadiation08
from dcbh_environment.dark_matter_halos import HaloModel

from multiprocessing import Pool


def monte_carlo_simulation(environment_params, mc_params, seed, proc_id):
    """This function sets up and runs a Monte Carlo simulation for sampling
    the Lyman-Werner flux seen by a central halo. The simulation is summarised
    by eqn 5 of Dijkstra et al 2008.
    
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
    
    halo_luminosity = LymanWernerRadiation08(cosmology, rng)
    halo_luminosity.init_luminosities(ms / cosmology.Hubble)
    
    # Here we unpack some Monte Carlo related parameters and open a file to
    # write results to.
    N_mc, epsilon = mc_params
    results = open(f'./samples_{proc_id}.txt', 'a')
    
    
    #======================== Monte Carlo Simulation =========================#
    
    ti = time.time()
    # Generate N_mc Monte Carlo samples.
    for sample in range(N_mc):
        J_LW = 0
        
        # Loop over mass/radius bins
        for mi in range(N_m):
            for rj in range(N_r):
                # Get number of halos in this bin
                num_halos = central_halo_env.num_halos(mi, rj)
                
                # Compute the LW flux from the star forming halos in this bin.
                if num_halos > 0:
                    area = central_halo_env.sphere_areas[rj]
                    if num_halos <= 10:
                        star_forming_halos = rng.binomial(num_halos, 1-epsilon)
                        L_LWs = halo_luminosity(mi, star_forming_halos)
                        J_LW += L_LWs.sum() / area
                        
                    else:
                        mean_L_LW = halo_luminosity[mi]
                        J_LW += num_halos * (1-epsilon) * mean_L_LW / area
        
        # Convert J_LW to 1e−21 erg / (s cm^2 sr Hz) and write result to disk.
        J_LW *= 1e21 * 1e26 / (4 * np.pi)
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
        z         = 10,
        Om0       = 0.27,
        Ode0      = 0.73,
        Ob0       = 0.042,
        H0        = 70,
        sigma_8   = 0.8344,
    )
    h = cosmology.Hubble

    M     = 4e7 * h              # Central halo mass in solar masses / h
    m_min = np.log10(4e7 * h)    # Minimum halo mass in log10 solar masses / h
    m_max = np.log10(1e14 * h)   # Maximum halo mass in log10 solar masses / h
    r_min = np.log10(0.002 * h)  # Minimum separation in physical Mpc / h
    r_max = np.log10(18  * h)    # Maximum separation in physical Mpc / h

    epsilon_dc = 0.1  # Duty cycle

    N_mc = 1000 # Number of Monte Carlo realisations
    N_m  = 500  # Number of mass bins
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
    
    # Create a pool of worker processes and start them working on MC sims.
    with Pool(num_procs) as pool:
        pool.starmap(monte_carlo_simulation, work)
    # monte_carlo_simulation(*work[0]) # for debugging