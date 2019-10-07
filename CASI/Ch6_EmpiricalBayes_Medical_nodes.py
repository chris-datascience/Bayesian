# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 21:52:36 2019

@author: Kris
"""


import numpy as np
from numpy.polynomial import polynomial
from scipy.optimize import minimize, fmin_slsqp
from scipy.stats import binom
import pandas as pd
import matplotlib.pyplot as plt

def binom_func():
    # TO DO: vectorised eval of binomial to get rid of loops
    pass

def lik_func(poly_coeff):
    log_likeh = 0  # (6.41)
    for _,row in Obs.iterrows():   #Obs.sample(20).iterrows():
        f_alpha = 0  # marginal_prob, eqn. (6.40)
        for theta in np.linspace(0,1,71): 
            rv = binom(row.n, theta)  # resp. number of trials and probability
            G_theta = polynomial.polyval(theta, poly_coeff)  # (6.38), log comes in later
            f_alpha += rv.pmf(row.x) * G_theta
        log_likeh += max(1e-50, np.log(f_alpha))  # preventing log of negative values..?!
    return -log_likeh


if __name__=='__main__':
    # Load observations (from CASI website)
    Obs = pd.read_csv('../../../data/nodes_CASI.txt', sep=' ')
    
    res = minimize(lik_func, (.1,.1,.1,.1,.1), options={'maxiter':5}, bounds=[(-10,10) for i in range(5)])#, constraints={'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] + x[4] - 1}) # maxiter=100, maxfun=100)
    print(res)
    
    plt.figure(figsize=(7,6))
    theta_ = np.linspace(0,1,31)
    plt.plot(theta_, polynomial.polyval(theta_, res.x), 'b-')
    
     