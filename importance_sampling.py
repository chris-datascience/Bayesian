# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 20:14:40 2020

@author: Kris
"""

"""
    Sources:    BDA3, Gelman et al. Chapter 10
                https://astrostatistics.psu.edu/su14/lectures/cisewski_is.pdf
                http://ib.berkeley.edu/labs/slatkin/eriq/classes/guest_lect/mc_lecture_notes.pdf
                https://towardsdatascience.com/importance-sampling-introduction-e76b2c32e744
"""

import numpy as np
from scipy.stats import norm, t
import matplotlib.pyplot as plt


def step(x, left, right):
    """ step interval function (Dirac), e.g. serving as h(x) in Importance sampling. """
    return (left <= x) * (x < right) * 1  # * 1/(right - left) 

def compute_IC_estimate_normal_proposal(SD=5):
    IC_estimates = []
    for _ in range(100):
        g_x = norm(loc=0, scale=SD)
        g_x_samples = g_x.rvs(size=N)
        g_x_samples_relevant = g_x_samples[g_x_samples < -2]
        IC_est = (1 / N) * np.sum(f_x.pdf(g_x_samples_relevant) / g_x.pdf(g_x_samples_relevant))
        IC_estimates.append(IC_est)
    print('\nIC estimate N(0,{}) = {}'.format(SD, np.mean(IC_estimates)))
    print('mean delta = %2.2f%%' % (100*np.abs(np.mean(IC_estimates) - Groundtruth) / Groundtruth))
    print('st.dev = %2.6f' % (np.std([x - Groundtruth for x in IC_estimates])))
    

if __name__=='__main__':
    
    N = 10000
    
    # =============================================================================
    # Example [1] Estimate cumul.density:  P(Y<-2) from Y ~ N(0,1)
    # Note that h(x) is a 0/1 step function (dirac delta) centered around -2
    # =============================================================================
    f_x = norm(loc=0, scale=1) # target dist
    Groundtruth = f_x.cdf(-2)
    print('Groundtruth = {}'.format(Groundtruth))
    
    # 1a Proposal dist. = N(0,5)    
    compute_IC_estimate_normal_proposal(5)
    
    # 1b Proposal dist. = N(0,2)
    compute_IC_estimate_normal_proposal(2)
    
    # 1c Proposal
    g_x = lambda x: step(x, -4, 4)
    IC_estimates = []
    for _ in range(100):
        g_x_samples = np.random.uniform(low=-4, high=4, size=N)
        g_x_samples_relevant = g_x_samples[g_x_samples < -2]
        IC_est = (1 / N) * np.sum(f_x.pdf(g_x_samples_relevant) / (1/8)*np.ones(len(g_x_samples_relevant))) # NB. Why is 1/8 constant necessary?
        IC_estimates.append(IC_est)
    print('\nIC estimate U(-4,4) = {}'.format(np.mean(IC_estimates)))
    print('mean delta = %2.2f%%' % (100*np.abs(np.mean(IC_estimates) - Groundtruth) / Groundtruth))
    print('st.dev = %2.6f' % (np.std([x - Groundtruth for x in IC_estimates])))
    
    
    plt.figure(figsize=(9,6))
    x_ = np.linspace(-5,5,1000)
    plt.plot(x_, f_x.pdf(x_), 'k-', lw=2, alpha=.5, label='target N(0,1)')
    plt.plot(np.linspace(-5,-2,100), f_x.pdf(np.linspace(-5,-2,100)), 'k-', lw=4)
    plt.plot(x_, norm(loc=0, scale=2).pdf(x_), 'b--', lw=1.5, alpha=.6, label='N(0,2')
    plt.plot(np.linspace(-5,5,1000), (1/8)*g_x(x_), 'r--', alpha=.7, label='local uniform')
    plt.legend()
    
    # =============================================================================
    #     Example [2] Estimate area under N(0,1), which is 1, using t-dist.
    #       See online Berkeley lecture notes
    # =============================================================================
    degrees_of_freedom = 1
    g_func = t(degrees_of_freedom)  # proposal distribution
    t_samples = g_func.rvs(N)    
    q_func = norm(loc=0, scale=1) # target function
    h_func = 1  # constant for all samples
    
    weights = q_func.pdf(t_samples) / g_func.pdf(t_samples)
    #IC_estimates = np.mean(h_func * weights) / np.mean(weights)
    plt.figure(figsize=(7,5))
    plt.hist(weights, density=True, label='dof=10')
    plt.vlines(np.mean(weights), 0, 2, 'k')
    plt.legend()

    # =============================================================================
    #     Example [2b] Now estimate actual pdf of N(0,1) using t(3),
    #       Exercise as proposed in DBA3 section 10.4
    # =============================================================================
    degrees_of_freedom = 3
    g_func = t(degrees_of_freedom)  # proposal distribution
    t_samples = g_func.rvs(N)    
    q_func = norm(loc=0, scale=1) # target function   
    h_func = step
    weights = q_func.pdf(t_samples) / g_func.pdf(t_samples)
    IC_estimate = []
    dx = .1  # to do: sensitivity analysis on dx relative to N
    for i,(x0,x1) in enumerate(zip(np.arange(-3,3,dx), np.arange(-3+dx, 3+dx, dx))):
        IC_estimate.append( np.mean(h_func(t_samples, x0, x1) * weights) / np.mean(weights) )
    
    IC_estimate = np.array(IC_estimate)
    IC_estimate /= np.sum(IC_estimate)
    target = q_func.pdf(np.arange(-3, 3, dx))
    target /= np.sum(target)
    plt.figure(figsize=(7,5))
    plt.plot(np.arange(-3, 3, dx), target, 'b--', lw=2.5, alpha=.3, label='N(0,1)')
    plt.plot(np.arange(-3, 3, dx), IC_estimate, 'g-', lw=1, alpha=.75, label='t (dof=3)')
    plt.xlabel('x', fontsize=14)
    plt.legend(fontsize=14)
    plt.title('Importance Sampling estimate of N(0,1) using t distribution', fontsize=15)

    # =============================================================================
    #     Example [3] Perform Bayesian inference for 1D non-hierarchical model
    #     Spse Yi ~ N(0,sigma2) where sigma2 ~ Exp(lambda)
    #     & Want to know posterior of lambda and Variance of marginal posterior of Y, say.
    # =============================================================================
    # to do
    
    # =============================================================================
    #     Example [4] Exercises 6 and 7 on pg. 273 of DBA3 Chapter 10
    # =============================================================================
    # to do