# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 15:22:29 2020

@author: Kris
"""

import numpy as np
import pandas as pd
import time
#from scipy.optimize import minimize
from scipy.stats import gamma as gamma_dist
from scipy.special import gamma
from scipy.stats import poisson, norm
import matplotlib.pyplot as plt
from math import factorial
import seaborn as sns
sns.set_style(style='dark')

"""
    Bayesian updates of Gamma distribution as likelihood;
    using update scheme from https://en.wikipedia.org/wiki/Gamma_distribution#Conjugate_prior
"""


def Gamma_dist(alpha, beta, x):
    """ PDF of gamma dist.
        alpha>0 is shape parameter, beta>0 is rate parameter
        follows https://en.wikipedia.org/wiki/Gamma_distribution
    """
    return (beta**alpha) * (x**(alpha - 1)) * np.exp(-beta * x) / gamma(alpha)
 
def Gamma_cumul_dist(alpha, beta, x):
    """ cumulative F(x) of gamma dist.
        alpha>0 is shape parameter, beta>0 is rate parameter
        follows https://en.wikipedia.org/wiki/Gamma_distribution
    """
    return None  # to do


if __name__=='__main__':
    # Fitting a 1D Gamma distribution (two parameters)
    # NB. Doesn't have proper conjugate prior!

#    beta_range = [0.5, 1, 2]
#    alpha_range = list(range(1, 6, 1))
#    x = np.linspace(0,20, 101)
#    for a in alpha_range:
#        plt.figure()
#        for b in beta_range:
#            plt.plot(x, Gamma_dist(a, b, x), 'b-', label=f'b={b}')
#        plt.legend()
#        plt.title(f'a = {a}', fontsize=15)
    

    # -- Bayesian updates assuming fixed alpha shape param --
    # TO DO
    
    # Fit MLE using scipy given fixed shape parameter
    # generate samples:
    a_ = 4
    b_ = 1
    samples = gamma_dist.rvs(a_, 0, b_, size=20)  # middle param is location = x shift = 0
    a_given, loc_given, b_retrieved = gamma_dist.fit(samples, fa=a_, floc=0)
    print(f'\nBeta target={b_}\nBeta retrieved={b_retrieved}')
    
    plt.figure()
    plt.hist(samples, density=True)    
    plt.plot(np.linspace(0,15,31), gamma_dist.pdf(np.linspace(0,15,31), a_, scale=b_), 'm-')


