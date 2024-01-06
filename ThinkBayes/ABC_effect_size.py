# -*- coding: utf-8 -*-
"""
Created on Wed May 27 22:22:44 2020

@author: erdbrca
"""


# =============================================================================
# Demo of effect size estimate using Approximate Bayesian Computation
#
# After Exercise 10-1 on page 119 of <Think Bayes> by Allen Downey
# =============================================================================

import os, sys
import numpy as np
import pandas as pd
from scipy.stats import expon, binom, norm
from itertools import product
import matplotlib.pyplot as plt
plt.style.use('bmh')

""" The idea of ABC here is that instead of using an exact likelihood of the observations,
    we compute the LogLikelihood of sample statistics (sample mean & stdev.), obviously under the
    assumption of normalityof both groups.
    Then, we sample from the joint posterior (mu,sigma) to compute a Monte Carlo estimate of the effect
    size, defined as the difference of the means divided by the mean standard deviation, in order to
    compare the two populations.
    See:    https://en.wikipedia.org/wiki/Effect_size
            <Think Bayes> textbook chapter 10
            https://www.leeds.ac.uk/educol/documents/00002182.htm
            
    Extensions:
        - Apply similar Bayesian estimate to effect size of association between categorical variables,
        as described on the wikipedia page. 
"""

def loglikel(obs, robust=False, n_steps=100):
    """ obs are non-nan observations of a single group in a 1D vector.
        robust is flag for robust location and scale estimates
    """
    n = len(obs)
    # compute sample statistics
    if robust:
        m = np.median(obs)
        s = np.median(np.abs(obs - m)) # median absolute deviation aka. MAD
    else:
        m = np.mean(obs)
        s = np.std(obs, ddof=1)
    
    # hypothesized value ranges of sample statistics (after page 114 of TB)
    mu_range = np.linspace(m - s, m + s, n_steps)
    sigma_range = np.linspace(.5*s, 1.5*s, n_steps)
    posterior = pd.DataFrame(columns=['mu', 'sigma', 'logl'])
    for i,hypo in enumerate(product(mu_range, sigma_range)):
        mu, sigma = hypo
        loglike = 0
        # compute logl of m given hypo (i.e. probability of observed sample mean given certain combination of mu, sigma)
        standard_error_mu = sigma / np.sqrt(n)
        rv = norm(loc=mu, scale=standard_error_mu)
        loglike = rv.logpdf(m)
        
        # compute logl of sigma given hypo
        standard_error_sigma = sigma / np.sqrt(2 * (n - 1))
        rv = norm(loc=sigma, scale=standard_error_sigma)
        loglike += rv.logpdf(s)
        
        # save
        posterior.loc[i,:] = {'mu':mu, 'sigma':sigma, 'logl':loglike}
    posterior['logl0'] = posterior['logl'] - posterior['logl'].mean()
    posterior['l'] = posterior['logl0'].map(lambda x: np.exp(x))
    posterior['l_norm'] = posterior['l'] / posterior['l'].sum()
    return posterior[['mu','sigma','l_norm']]

def effect_size(mu1, mu2, sigma1, sigma2):
    return (mu1 - mu2) / ((sigma1 + sigma2) / 2)


if __name__=='__main__':

    # Sample two groups of data
    H0 = 178 + 7.7*np.random.randn(1000,) # heights of men
    H1 = 163 + 7.3*np.random.randn(700,)  # heights of women
        
    post0 = loglikel(H0[:])
    mode0 = post0.loc[post0.l_norm.idxmax(), ['mu','sigma']]
    
    post1 = loglikel(H1[:])
    mode1 = post1.loc[post1.l_norm.idxmax(), ['mu','sigma']]  
    
    # Estimate effect size
    # --------------------
    n_samples = 1000
    # [i]  Sample from mu,sigma-posterior of group 0
    sample_inds0 = np.random.choice(np.arange(len(post0)), size=n_samples, replace=True, p=post0.l_norm.values)
    
    # [ii] Sample from posterior group1
    sample_inds1 = np.random.choice(np.arange(len(post1)), size=n_samples, replace=True, p=post1.l_norm.values)
    
    # [iii] Combine to create effect size samples
    ES = np.empty(n_samples,)
    for i in range(n_samples):
        mu0 = post0.loc[sample_inds0[i], 'mu']
        sigma0 = post0.loc[sample_inds0[i], 'sigma']
        mu1 = post1.loc[sample_inds1[i], 'mu']
        sigma1 = post1.loc[sample_inds1[i], 'sigma']
        ES[i] = effect_size(mu0, mu1, sigma0, sigma1)
    plt.figure(figsize=(7,6))
    plt.hist(ES, bins=35, density=True, alpha=.5)
    plt.title('Posterior of Effect Size of heigths of men vs. women')
    
