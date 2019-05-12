# -*- coding: utf-8 -*-
"""
Created on Sun May 27 12:12:33 2018

@author: Kris
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, poisson, nbinom
#import math  # for math.factorial()
from scipy.special import binom
#import seaborn as sns

colors = 'brkgmyc'*10

def plot_expected_values(Exp):
    plt.figure(figsize=(8,5))
    plt.plot([0,len(Exp)-1], [.5, .5], 'k:')
    plt.plot(list(range(len(Exp))), Exp, 'ko-')
    plt.ylim(0, np.max(Exp))
    plt.title('Expected value of posterior (heads) after each new observation')

def plot_var(Var):
    plt.figure(figsize=(8,5))
    plt.plot(list(range(len(Var))), Var, 'co-')
    plt.ylim(0,np.max(Var))
    plt.title('Variance of coin bias')

def Gamma_plot(alpha=2, beta=2):
    loc = 0  # assumed fixed! loc=0 on wikipedia's Gamma distribution plots
    alpha = float(alpha)
    theta = 1. / float(beta)
    data = gamma.rvs(alpha, loc=loc, scale=theta, size=10000)
    x = np.linspace(0,10,250)
    plt.figure(figsize=(7,5))
    plt.hist(data, bins=55, density=True)
    plt.plot(x, gamma.pdf(x, alpha, loc=loc, scale=theta), 'r-', linewidth=3)
    plt.title('GAMMA DISTRIBUTION with Scipy')  
    print(np.mean(data))

def Poisson_plot(lambd=2.3, kmax=18, label='Poisson'):
    x = np.arange(0,kmax)
    plt.figure(figsize=(9,6))
    plt.vlines(x, ymin=np.zeros_like(x), ymax=poisson.pmf(x, mu=lambd), colors='r', lw=4, linestyles='solid', label=label)
    #sns.barplot(np.arange(3), [1,2,2])  # alternative
#    plt.title('POISSON DISTRIBUTION with Scipy')

#def combination_coeff(n, k):
#    return math.factorial(n) / (math.factorial(n - k) * math.factorial(k))

def neg_Binom_convert_params(mu, var):
    return (NB_var - NB_mean) / NB_var, NB_mean**2 / (NB_var - NB_mean)

def neg_binom(p, r, k):
    """
    Negative Binomial using p and r as parameters, k as indep.var.
    p is prob of success  (float, 0 < x < 1)
    r is number of failures  (int)
    k is is number of successes (int)
    Computed is P(X=k)
    See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
    """
    binom_coeff = binom(k + r - 1, k)
    return binom_coeff * ((1 - p)**r) * (p**k)        
    
def neg_binom_pmf(p, r, k_max=15):
    pmf = np.empty((k_max + 1))
    for k in range(k_max+1):
        pmf[k] = neg_binom(p, r, k)
    return pmf


if __name__=='__main__':
    """
    Poisson likelihood with Gamma prior.
    Negative Binomial predictive posterior
    """
    
    # Plotting Demo
    # =============
    Gamma_plot(2, 20)
#    Poisson_plot()
    
    # Bayesian inference
    # ==================
    observations = np.array([1,4,2,2,2,0,12,7])  #  delay days
    #SQRTobs = np.sqrt(observations)
    n_obs = len(observations)  # number of observations so far (=len(y))
    alpha0, beta0 = 2, 2  # assumed prior Gamma distribution
    prior_Poisson_lambd = alpha0 / beta0 # expectation of Gamma

    # Analytical solution of Predictive Posterior is Negative Binomial
    # Sources:
    #    http://people.stat.sc.edu/Hitchcock/stat535slidesday18.pdf
    #    https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
    NB_mean = (np.sum(observations) + alpha0) / (n_obs + beta0)
    NB_var = ((np.sum(observations) + alpha0) * (n_obs + beta0 + 1)) / (n_obs + beta0)**2
    NB_p, NB_r = neg_Binom_convert_params(NB_mean, NB_var)
    
    kmax = 15
    posterior_pmf = neg_binom_pmf(NB_p, NB_r, kmax)  # computing posterior pmf
    
    x = np.arange(0, kmax+1)
    PMFs = pd.DataFrame(index=x, columns=['prior', 'posterior'])
    PMFs['prior'] = poisson.pmf(x, mu=prior_Poisson_lambd)
    PMFs['posterior'] = posterior_pmf
    PMFs.plot.bar(figsize=(11,8))
#    Poisson_plot(lambd=prior_Poisson_lambd, kmax=kmax, label='prior')
#    plt.vlines(x, 0, posterior_pmf, colors='b', lw=4, linestyles='solid', label='posterior')
#    plt.legend(fontsize=15)

#    plt.vlines(x, 0, nbinom.pmf(x, n, p), colors='b', lw=4, alpha=0.5)