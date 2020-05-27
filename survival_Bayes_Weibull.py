# -*- coding: utf-8 -*-
"""
Created on Sun May 24 12:45:22 2020

@author: erdbrca
"""


# https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Piecewise%20Exponential%20Models%20and%20Creating%20Custom%20Models.html#Discrete-survival-models

# =============================================================================
# Monitor exponential delays in one-event survival data over time
# =============================================================================


import os, sys
import numpy as np
import pandas as pd
from scipy.stats import expon, binom
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
from lifelines import WeibullFitter, ExponentialFitter, PiecewiseExponentialFitter
from lifelines.datasets import load_waltons
# from lifelines.utils import datetimes_to_durations, median_survival_times
import matplotlib.pyplot as plt
#from exponential_post_predictive import Bayesian_conjugate_inference
#from decay_problem_MacKay import expon_integral

plt.style.use('bmh')


def weibull_pdf(t, lambda_, rho_):
    """ Same parametrisation as in 
        https://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html
    """
    return (rho_ / (lambda_**rho_)) * (t**(rho_ - 1)) * np.exp(-(t / lambda_)**rho_)

def weibull_survival(t, lambda_, rho_):
    return np.exp(-(t / lambda_)**rho_)


if __name__=='__main__':
    
    # =============================================================================
    #     Example dataset from Lifelines
    #     https://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html
    # =============================================================================
    df = load_waltons()
    
    T = df['T']
    E = df['E']
    # kmf = KaplanMeierFitter()
    # kmf.fit(T, event_observed=E)

    wf = WeibullFitter().fit(T, E)
    
    # kmf.plot()
    # kmf.cumulative_density_.plot(figsize=(7,6))

    naf = NelsonAalenFitter()
    naf.fit(T,event_observed=E)
    plt.figure(figsize=(8,6))
    naf.plot()
    wf.plot()
    plt.title('cumulative hazard (Waltons dataset)')
    
    print('fitted Weibull parameters (MLE):')
    print('\tlambda = {}'.format(wf.lambda_))
    print('\trho = {}'.format(wf.rho_))
    
    # =============================================================================
    #     Bayesian model estimate of same Weibull params
    # =============================================================================
    print('\nBayesian estimates (flat priors):') # to do: apply actual sensible priors
    lam_range = np.linspace(1, 100, 50)
    rho_range = np.linspace(0.1, 6, 30)
    L, R = np.meshgrid(lam_range, rho_range)
    prior = np.ones_like(L)
    prior /= np.sum(prior, axis=(0,1))
    logprior = np.log(prior)
    logprior /= np.sum(logprior)
    
    # In log dimension (check: still dangerously small numbers?)
    logpost = logprior[:]
    for _,row in df.iterrows():
        for i,rho_ in enumerate(rho_range):
            for j,lambda_ in enumerate(lam_range):
                if row.E==1:
                    logpost[i,j] += np.log(weibull_pdf(row['T'], lambda_, rho_))
                else:
                    logpost[i,j] += np.log(weibull_survival(row['T'], lambda_, rho_))
    maxlogl = np.max(logpost)
    post = np.exp(logpost - maxlogl) # shift by max.likel. to reduce underflow
    post /= np.sum(post, axis=(0,1))

    # Plot joint post.dist.
    plt.figure(figsize=(8,6))
    plt.contourf(L, R, post, alpha=.9)
    plt.plot(wf.lambda_, wf.rho_, 'ro', markersize=12, label='MLE')
    plt.title('Joint posterior')
    plt.xlabel('lambda')
    plt.ylabel('rho')
    plt.xlim(40,60)
    plt.ylim(3,4)
    plt.legend()
    
    # Lambda marginal post.
    lam_max_ind = np.argmax(post.max(axis=0))
    lambda_max = lam_range[lam_max_ind]  # mode
    lambda_EV = np.dot(lam_range, post.max(axis=0)/post.max(axis=0).sum()) # expected value
    print('Mean of lambda posterior = {}'.format(lambda_EV))
    plt.figure(figsize=(7,6))    
    #plt.plot(lam_range, post.max(axis=0)/post.max(axis=0).sum(), label='argmax')
    plt.plot(lam_range, post.mean(axis=0)/post.mean(axis=0).sum(), label='mean')
    plt.title('lambda marginal posterior')
    plt.xlabel('lambda')
    plt.legend()

    # Rho marginal post.
    rho_EV = np.dot(rho_range, post.max(axis=1)/post.max(axis=1).sum()) # expected value
    print('Mean of rho posterior = {}'.format(rho_EV))
    plt.figure(figsize=(7,6))    
    plt.plot(rho_range, post.mean(axis=1)/post.mean(axis=1).sum(), label='mean')
    plt.title('rho marginal posterior')
    plt.xlabel('rho')
    plt.legend()

