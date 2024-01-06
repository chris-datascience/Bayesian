# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:56:33 2020

@author: Kris
"""

import numpy as np
import pandas as pd
#from scipy.optimize import minimize
#from scipy.special import gamma
from scipy.stats import poisson, binom
import matplotlib.pyplot as plt
from math import factorial
import time
import seaborn as sns
sns.set_style(style='dark')

def n_combinations(a, b):
    """ (A B):= A! / B!(A-B)! i.e. A >= B required """
    return int(factorial(a) / (factorial(b) * factorial(a - b)))

def inverse_cumulative_sampling(pmf, n_samples):
    """ Generic function.
        pmf is a discrete ditribution;
        type is a pd.Series with index containing the distribution's values
          and the values of the series containing the probabilities (summing up to one).
    """
    pmf /= pmf.sum()
    sample_inds = np.searchsorted(pmf.cumsum().values, np.random.uniform(size=n_samples), side='left')
    return pmf.index[sample_inds].values

def make_flat_prior(x_min, x_max, n_pts):
    prior_pmf = np.linspace(x_min, x_max, n_pts)
    prior_pmf /= prior_pmf.sum()
    return prior_pmf
    
def Poiss(lambd, k):
    #return (np.exp(-lambd) * lambd**k) / factorial(k)
    #rv = poisson(lambd)
    #return rv.pmf(k)
    return poisson.pmf(lambd, k)

def Binomial(n, p, k):
    #return n_combinations(n, k) * (p ** k) * ((1 - p) ** (n - k))
    return binom.pmf(k, n, p)


if __name__=='__main__':
    # Settings
    f = .10  # assumed known, so fixed
    k_obs = 15  # observation
    r_range = np.arange(1,1001) #np.linspace(0, 400, 300)
    r_prior = make_flat_prior(1, 500, 1000) # unspecified in textbook, take naive informative
    n_range = np.arange(k_obs, 750)  # note that n<k_obs is useless!
    color = 'kgrbycm'
    
    # Reproduce figure 14-1 (by means of validation)
    plt.figure()
    for i,r in enumerate([100, 250, 400]):
        posterior_n = Poiss(r, n_range) * Binomial(n_range, f, k_obs)
        posterior_n /= np.sum(posterior_n)
        plt.plot(n_range, posterior_n, color[i]+'-', label='r = '+str(r))
    plt.legend(fontsize=14)
    plt.title('Figure 14-1', fontsize=15)
    plt.xlabel('n', fontsize=14)
    plt.ylabel('P(n | y)', fontsize=14)
    
    # Now for full range of r values, and computing posterior of r as well (by making mixture pmf)
#    # [1] Explicit, easy-to-read, slow version
#    t0 = time.time()
#    r_posterior = np.zeros((len(r_range),), dtype=float)
#    for j,r in enumerate(r_range):
#        r_post_mix = np.zeros((len(n_range),), dtype=float)
#        # Compute P(n | y) for single value of r
#        posterior_n = np.empty((len(n_range),), dtype=float)
#        for i,n in enumerate(n_range):
#            posterior_n[i] = Poiss(r, n) * Binomial(n, f, k_obs)
#            #posterior_n[i] = np.log(Poiss(r, n)) + np.log(Binomial(n, f, k_obs))
#            #posterior_n /= np.sum(posterior_n)
#            
#            # Compute P(r | n) as a mixture of Poissons given posterior of n [see "make_mixtures" post]
#            #n_pmf = zip(n_range, posterior_n)
#            #r_post_mix = 0
#            #for n_val, n_prob in n_pmf:
#            #    r_post_mix += n_prob * Poiss(r, n_val)
#            
#            r_post_mix[i] += posterior_n[i] * Poiss(r, n)
#        #r_posterior[j] = r_post_mix / r_post_mix.sum()

    # [2] Faster, largely unlooped version [esp. vectorised in n]
    # ---------------------------------------------------------------
    t0 = time.time()
    #r_posterior = np.zeros((len(r_range),), dtype=float)
    n_likelihood = Binomial(n_range, f, k_obs)
    r_post_mix = np.zeros((len(n_range), len(r_range)), dtype=float)  
    for j,r in enumerate(r_range):
        # (i) Compute P(n | y) for single value of r (and for entire range of n values)
        posterior_n = Poiss(r, n_range) * n_likelihood
        
        # (ii) Compute P(r | n) as a mixture of Poissons given posterior of n [see "make_mixtures" post]
        r_post_mix[:,j] = posterior_n * Poiss(r, n_range)  # store meta-pmf times likelihood of r
    
    # (iii) Distill mixture into single pmf & mulitply by prior
    r_posterior = r_post_mix.sum(axis=0) * r_prior
    r_posterior /= np.sum(r_posterior)
    print('\nVectorised computation finished in %2.3f seconds.' % (time.time() - t0))
    # (iv) Compute final P(n|k_obs) as mixture; This uses r posterior P(r|n) as prior distribution for n
    n_final_posterior_mix = r_posterior.reshape(-1,1) * Poiss(r_range.reshape(-1,1), n_range.reshape(1,-1))
    n_final_posterior = n_final_posterior_mix.sum(axis=0) #/ n_final_posterior_mix.sum(axis=(0,1))

    
    plt.figure(figsize=(9,6))
    plt.plot(r_range, r_posterior, lw=2, alpha=.7, label='posterior of r')
    plt.plot(n_range, n_final_posterior, '--', lw=2, alpha=.8, label='posterior of n')
    plt.legend(fontsize=14)
    plt.xlabel('emission rate', fontsize=14)
    plt.title('Figure 14-2', fontsize=15)

    # [3] Fully unlooped (broadcasted) version [which is void of any educational power]
    # TO DO: generalise for multiple-level hierarchy, using functools.reduce
    # -----------------------------------------------------------------------------------
    t0 = time.time()
    likelihood_r = Poiss(r_range.reshape(-1,1), n_range.reshape(1,-1))
    likelihood_n = Binomial(n_range.reshape(1,-1), f, k_obs)
    Likelihoods = [likelihood_r, likelihood_r, likelihood_n]
    #r_post_mix = likel_r**2 * Binomial(n_range.reshape(1,-1), f, k_obs)
    #r_posterior = r_post_mix.sum(axis=1) * r_prior
    #r_posterior = np.einsum('ij,ij->i', likelihood_r**2, likelihood_n) # multiply and sum over r range; no prior applied (i.e. fully flat prior)
    r_posterior = np.einsum('ij,ij,ij->i', *Likelihoods) # multiply and sum over r range; no prior applied (i.e. fully flat prior)
    r_posterior /= np.sum(r_posterior)
    print('\nBroadcasted computation finished in %2.3f seconds.' % (time.time() - t0))    

    plt.figure(figsize=(7,5))    
    plt.plot(r_range, r_posterior, lw=3, alpha=.7, label='posterior of r')
    plt.xlabel('r', fontsize=14)
    plt.legend(fontsize=14)
    plt.title('Broadcasted', fontsize=15)
    