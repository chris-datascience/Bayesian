# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:07:40 2019

@author: erdbrca
"""

"""
    Bayesian AB-testing in PyMC3.
    Template for comparing frequencies of e.g. conversions versus total number of visitors.
    Adapted from <<BMH>> textbook.
    See 
        https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter2_MorePyMC/Ch2_MorePyMC_PyMC3.ipynb
"""

import pymc3 as pm
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

run_models = {2}

if 1 in run_models:
    # The parameters are the bounds of the Uniform.
    with pm.Model() as model:
        p = pm.Uniform('p', lower=0, upper=1)
        
    #set constants
    p_true = 0.05  # remember, this is unknown.
    N = 1500
    
    # sample N Bernoulli random variables from Ber(0.05).
    # each random variable has a 0.05 chance of being a 1.
    # this is the data-generation step
    occurrences = stats.bernoulli.rvs(p_true, size=N)
    
    print(occurrences) # Remember: Python treats True == 1, and False == 0
    print(np.sum(occurrences))
    # Occurrences.mean is equal to n/N.
    print("What is the observed frequency in Group A? %.4f" % np.mean(occurrences))
    print("Does this equal the true frequency? %s" % (np.mean(occurrences) == p_true))
    
    #include the observations, which are Bernoulli
    with model:
        obs = pm.Bernoulli("obs", p, observed=occurrences)
        # To be explained in chapter 3
        step = pm.Metropolis()
        #trace = pm.sample(18000, step=step)
        trace = pm.sample(6000, step=step, tune=1000, cores=1)
        burned_trace = trace[1000:]
        
    # -- Plot posterior of unknown p_A: --
    plt.figsize(12.5, 4)
    plt.title("Posterior distribution of $p_A$, the true effectiveness of site A")
    plt.vlines(p_true, 0, 90, linestyle="--", label="true $p_A$ (unknown)")
    plt.hist(burned_trace["p"], bins=25, histtype="stepfilled", normed=True)
    plt.legend();

if 2 in run_models:
    # These two quantities are unknown to us.
    true_p_A = 0.05
    true_p_B = 0.04
    
    # Notice the unequal sample sizes -- no problem in Bayesian analysis.
    N_A = 1500
    N_B = 750
    
    # Generate some observations..
    observations_A = stats.bernoulli.rvs(true_p_A, size=N_A)
    observations_B = stats.bernoulli.rvs(true_p_B, size=N_B)
    print("Obs from Site A: ", observations_A[:30], "...")
    print("Obs from Site B: ", observations_B[:30], "...")
    print(np.mean(observations_A))
    print(np.mean(observations_B))
    
    # Set up the pymc3 model. Again assume Uniform priors for p_A and p_B.
    with pm.Model() as model:
        p_A = pm.Uniform("p_A", 0, 1)
        p_B = pm.Uniform("p_B", 0, 1)
        
        # Define the deterministic delta function. This is our unknown of interest.
        delta = pm.Deterministic("delta", p_A - p_B)
        
        # Set of observations, in this case we have two observation datasets.
        obs_A = pm.Bernoulli("obs_A", p_A, observed=observations_A)
        obs_B = pm.Bernoulli("obs_B", p_B, observed=observations_B)
    
        # To be explained in chapter 3.
        step = pm.Metropolis()
        #trace = pm.sample(20000, step=step)
        trace = pm.sample(18000, step=step, tune=1000, cores=1)
        burned_trace=trace[1000:]
        
    p_A_samples = burned_trace["p_A"]
    p_B_samples = burned_trace["p_B"]
    delta_samples = burned_trace["delta"]
    
    # POST: Histograms of posteriors
    plt.figure(figsize=(12.5, 10))
    ax = plt.subplot(311)
    plt.xlim(0, .1)
    plt.hist(p_A_samples, histtype='stepfilled', bins=25, alpha=0.85,
             label="posterior of $p_A$", color="#A60628", normed=True)
    plt.vlines(true_p_A, 0, 80, linestyle="--", label="true $p_A$ (unknown)")
    plt.legend(loc="upper right")
    plt.title("Posterior distributions of $p_A$, $p_B$, and delta unknowns")
    
    ax = plt.subplot(312)
    plt.xlim(0, .1)
    plt.hist(p_B_samples, histtype='stepfilled', bins=25, alpha=0.85,
             label="posterior of $p_B$", color="#467821", normed=True)
    plt.vlines(true_p_B, 0, 80, linestyle="--", label="true $p_B$ (unknown)")
    plt.legend(loc="upper right")
    
    ax = plt.subplot(313)
    plt.hist(delta_samples, histtype='stepfilled', bins=30, alpha=0.85,
             label="posterior of delta", color="#7A68A6", normed=True)
    plt.vlines(true_p_A - true_p_B, 0, 60, linestyle="--",
               label="true delta (unknown)")
    plt.vlines(0, 0, 60, color="black", alpha=0.2)
    plt.legend(loc="upper right");
    
    # Count the number of samples less than 0, i.e. the area under the curve
    # before 0, represent the probability that site A is worse than site B.
    print("Probability site A is WORSE than site B: %.3f" % \
        np.mean(delta_samples < 0))
    print("Probability site A is BETTER than site B: %.3f" % \
        np.mean(delta_samples > 0))