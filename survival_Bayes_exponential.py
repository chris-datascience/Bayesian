# -*- coding: utf-8 -*-
"""
Created on Sun May 24 12:45:22 2020

@author: erdbrca
"""

# =============================================================================
# Fitting Exponential delays in single event data
# =============================================================================


import os, sys
import numpy as np
import pandas as pd
from scipy.stats import expon, binom
from scipy.interpolate import interp1d, Akima1DInterpolator, UnivariateSpline
import matplotlib.pyplot as plt
plt.style.use('bmh')

from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
from lifelines import ExponentialFitter #WeibullFitter, PiecewiseExponentialFitter
from lifelines.datasets import load_waltons, load_nh4, load_rossi, load_stanford_heart_transplants

from survival_Bayes_generate_data import generate_constant_hazard


def bayesian_model_estimation(T, E, iter_interpolate=2, n_pts=20):
    """ T is durations
        E is binary event flag
        iter_interpolate is number of iterations in posterior grid interpolation refinement (int, min.=1)
        n_pts is number of points in posterior
    """
    # Plot non-parametric curves
    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=E)
    kmf.plot()
    # kmf.cumulative_density_.plot(figsize=(7,6))

    naf = NelsonAalenFitter()
    naf.fit(T,event_observed=E)
    plt.figure(figsize=(7,6))
    naf.plot()
    plt.title('Cumulative hazard rate')

    # Fit exponential cumulative hazard model
    exf = ExponentialFitter().fit(T, E, label='ExponentialFitter') #  See https://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html
    exf.plot_cumulative_hazard()
    print('fitted lambda = {}'.format(1 / exf.lambda_))      # Confidence bounds on this?  --> bootstrap?

    # Plot groundtruth curve
    plt.figure(figsize=(7,6))
    x = np.arange(1,30)
    plt.plot(x, expon(scale=1/target_rate).sf(x), 'g--', lw=2.5, alpha=.6, label='target')
    plt.plot(x, expon(scale=exf.lambda_).sf(x), 'r-',lw=3, alpha=.7, label='fitted')
    plt.legend()
    plt.xlabel('duration (time since event arrival')
    plt.title('Survival curve')
    
    # Bayesian inference of lambda
    # ============================
    lam_range = np.linspace(0, .2, n_pts)
    for it in range(1,iter_interpolate+1):
        print('\niteration {}'.format(it))
        prior = np.ones_like(lam_range)
        prior /= np.sum(prior)
        logprior = np.log(prior)
        logprior /= np.sum(logprior)
        
        # Compute likelihood in original dimension (dangerously small numbers!)
        # post = prior
        # for duration, event_flag in zip(T, E):
        #     if event_flag==1:
        #         post *= expon(scale=1/lam_range).pdf(duration)
        #     else:
        #         post *= expon(scale=1/lam_range).sf(duration)     
        
        # Compute likelihood in log dimension
        logpost = logprior #- lam_range*T.sum() + np.log(lam_range)*(1 - E).sum() # <-- vector implentation is wrong
        for duration, event_flag in zip(T, E):
            if event_flag==1:
                logpost += expon(scale=1/lam_range).logpdf(duration)
            else:
                logpost += expon(scale=1/lam_range).logsf(duration)
        # Trick: shift entire log dist. by max.loglikel. before exponentiation to reduce potential underflow:
        maxlogl = np.max(logpost)
        post = np.exp(logpost - maxlogl)
        post /= np.sum(post)
        ExpectedVal = np.dot(lam_range, post)
        print('Mean of lambda posterior = {}'.format(ExpectedVal))
        print('MAE = {}'.format(np.abs(ExpectedVal - target_rate)))
        
        # Plot lambda posterior
        plt.figure(figsize=(7,6))
        plt.plot(lam_range, post, 'b.-', lw=1, label='Bayes')
        plt.vlines(1 / exf.lambda_, 0, 1.2*np.max(post), color='m', lw=3, alpha=.6, label='MLE')
        plt.vlines(target_rate, 0, 1.2*np.max(post), color='orange', lw=3, alpha=.9, label='target')
        plt.vlines(ExpectedVal, 0, 1.2*np.max(post), color='b', lw=3, alpha=.6, label='Bayes EV')
        plt.legend()
        plt.title('Lambda estimate (iteration {})'.format(it))
        plt.xlabel('lambda')
        
        # Refine posterior grid evaluation points
        if it<=iter_interpolate:        
            cumul_prob_dens = post.cumsum()
            f = interp1d(cumul_prob_dens, lam_range)
            cdf_new_grid_pts = np.linspace(1e-2, 1 - 1e-2, n_pts)
            lam_range = f(cdf_new_grid_pts)


if __name__=='__main__':

    # [i] On artificially generated data:
    # ===================================
    target_rate = .1
    sim_durations = generate_constant_hazard(target_rate=target_rate)
    bayesian_model_estimation(sim_durations['duration'], sim_durations['event_flag'], \
                              iter_interpolate=5, n_pts=20)
    
    
    # [ii] On Rossi dataset:
    # ======================
    # Source: https://lifelines.readthedocs.io/en/latest/lifelines.datasets.html
    # df = load_rossi()
    # bayesian_model_estimation(df['week'], df['arrest'])
   
    # TO DO