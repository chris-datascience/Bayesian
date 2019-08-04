# -*- coding: utf-8 -*-
"""
Created on Sun May 27 17:21:40 2018

@author: Kris
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm, nbinom

from thinkbayes3 import Pmf, Suite
#from thinkbayes3_official import MakeMixture

def MakeGaussianPmf(mu, sigma, num_sigmas, n=101):
    pmf = Pmf()
    low = mu - num_sigmas*sigma
    low = max(low, 0)
    high = mu + num_sigmas*sigma
    for x in np.linspace(low, high, n):
        pmf.Set(x, norm.pdf(x, loc=mu, scale=sigma))
    pmf.Normalize()
    return pmf

def EvalPoissonPmf(lambda_, k):
    """
    lambda_>0 is shape parameter
    k is number of occurrences, see wiki on Poisson distribution
    """
    return (lambda_)**k * np.exp(-lambda_) / math.factorial(k)

def EvalExponentialPdf(lambda_, x):
    return lambda_ * np.exp(-lambda_ * x)

def EvalNegBinomial(n, p, x):
    return nbinom.pmf(x, n, p)

def MakePoissonPmf(lambda_, high):
    pmf = Pmf()
    for k in range(0, high+1):
        p = EvalPoissonPmf(lambda_, k)
        pmf.Set(k, p)
    pmf.Normalize()
    return pmf

class TransactionFrequency(Suite):
    def __init__(self, mu_prior, sigma_prior):
        pmf = MakeGaussianPmf(mu_prior, sigma_prior, 4)
        Suite.__init__(self, pmf)

    def Likelihood(self, data, hypothesis):
        lambda_ = hypothesis  # hypothetical value of parameter
        k = data # observation        
        return EvalPoissonPmf(lambda_, k)

def computeTransFrequencyPmF(suite, upper_bound_txs=20):
    """
    This reconstructs the distribution of the transactions per minute from 
    a distribution of the lambda parameter of a Poisson process.
    See Chapter 7, pg. 68 of <<Think Bayes>>.
    """
    metapmf = Pmf()
    for lambda_, prob in suite.Items():
        pmf = MakePoissonPmf(lambda_, upper_bound_txs) # creates a pmf for each 'hypothetical' lambda value
        metapmf.Set(pmf, prob)
    return MakeMixture(metapmf)
    
if __name__=='__main__':
    """
    Case 2: Number of card transactions in a certain time period
    This is related to the time between processed transactions (but easier to model).
    Poisson likelihood, gaussian prior.
    Note: we are not using conjugate priors anywhere here! Instead we discretise the prior.
    Based on chapter 7 of <<Think Bayes>> (Allen Downey).
    Pmf means probability mass function
    """
    
    # Initiation
    mu_prior, std_prior = 6, 3
    suite1 = TransactionFrequency(mu_prior, std_prior)  # initiate with assumed mu, sigma of Gaussian prior
    
    # Process new observations and compute posterior
    observed = [0,0,3,5,14,3,2,5,8,2,2,0,0,1]+[14]*13
    suite1.UpdateSet(observed)  # Update using observations of times between transactions (in minutes or whatever)
    
    # Plot prior and posterior in one
    MAP = suite1.MaximumLikelihood()
    Max_like = suite1.MaxLike()
    lambda_hypo = np.array([a for a,b in suite1.Items()])
    prob_density = np.array([b/Max_like for a,b in suite1.Items()])
    print('sum prob. density = %.2f' % np.sum(prob_density))
    
    plt.figure(figsize=(9,7))
    plt.plot(lambda_hypo, norm.pdf(lambda_hypo, loc=mu_prior, scale=std_prior), 'r-', label='prior')
    plt.plot(lambda_hypo, prob_density, 'b-', label='posterior')
    plt.legend(fontsize=14)
    plt.xlabel('lambda', fontsize=14)
    
    # Compute and plot distribution of number of transactions per time period
    """
    Note: because we don't know parameter lambda exactly, we need to create a 
    mixture Poisson distribution looping over the posterior distribution of lambda
    --> SOMEHOW CODE IS BROKEN!
    """
#    txs_per_time_unit_Pmf = computeTransFrequencyPmF(suite1, upper_bound_txs=20)
#    plt.figure(figsize=(9,7))
#    plt.plot([a for a,b in txs_per_time_unit_Pmf.Items()], [b for a,b in txs_per_time_unit_Pmf.Items()], 'g-')
    
    # Compute pdf for single lambda:
    cdf = np.cumsum(prob_density)/np.cumsum(prob_density)[-1]
    print('cdf max = %.2f' % np.max(cdf))
    
    lambda_crit = lambda_hypo[cdf >= .9][1] # 90-percentile
    tx_freq_dist = MakePoissonPmf(lambda_crit, max(observed)+10)
    
    plt.figure(figsize=(9,7))
    tx_freq_probs = [b for a,b in tx_freq_dist.Items()] # tx_freq_dist.Probs(Values())
    plt.bar([a for a,b in tx_freq_dist.Items()], tx_freq_probs)
    print('integral of pdf of 90%% lambda = %.2f' % np.sum(tx_freq_probs))
    # TO DO: define critical lambda
    # How to deal with zeros in Poisson?