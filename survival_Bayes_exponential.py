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
from lifelines.utils import datetimes_to_durations, median_survival_times
import matplotlib.pyplot as plt
#from exponential_post_predictive import Bayesian_conjugate_inference
#from decay_problem_MacKay import expon_integral

plt.style.use('bmh')


def expon_pdf(y, lambd):
    return lambd * np.exp(-lambd * y)

def expon_ccdf(y, lambd):
    return np.exp(-lambd * y)


if __name__=='__main__':
    
    # =============================================================================
    #     Example dataset from Lifelines
    # =============================================================================
    # df = load_waltons()
    # dfC = df[df.group=='control'].copy()
    
    # T = dfC['T']
    # E = dfC['E']
    # kmf = KaplanMeierFitter()
    # kmf.fit(T, event_observed=E)

    # kmf.plot()
    # kmf.cumulative_density_.plot(figsize=(7,6))
    
    # naf = NelsonAalenFitter()
    # naf.fit(T,event_observed=E)
    # print(naf.cumulative_hazard_.head())
    # naf.plot()

    # Generate dataset
    # ----------------
    n_events = 100
    t_max = 40
    target_rate = .1
    # noise_factor = .05
    #rate = target_rate * np.ones((t_max,)) + noise_factor * np.random.randn(t_max)  # as function of discrete time variable!
    rate = target_rate # constant hazard rate (independent of t)
    obs = pd.DataFrame()
    obs.index.name = 't'
    events_left = n_events
    t = 0
    while events_left>np.ceil(1/rate): #all([events_left>0, t<t_max-1]):
        obs.loc[t,'unconverted_events'] = events_left
        new_events = int(np.round(rate * events_left))
        obs.loc[t,'events'] = new_events
        events_left -= new_events
        t += 1
    # Add extra
    if t<t_max:
        events_to_add = binom(t_max - t, rate).rvs(1)
        for e in events_to_add:
            obs.loc[t, 'unconverted_events'] = events_left
            obs.loc[t, 'events'] = 1
            events_left -= new_events
            t += 1  
    # Convert to durations & event flags
    observed_freq = obs[['events']].reset_index().astype(int)
    observed_freq['freq'] = observed_freq['events'].shift(1)
    observed_freq = observed_freq[1:].copy()
    observed_durations = pd.DataFrame(columns=['duration', 'event_flag'])
    ind = 0
    for _,row in observed_freq.iterrows():
        for __ in range(int(row.freq)):
            observed_durations.loc[ind, 'duration'] = row.t
            observed_durations.loc[ind, 'event_flag'] = 1
            ind += 1
    for __ in range(events_left):
        observed_durations.loc[ind, 'duration'] = t
        observed_durations.loc[ind, 'event_flag'] = 0
        ind += 1
    observed_durations = observed_durations.astype(int)

    # Plot non-parametric curves
    # --------------------------
    T = observed_durations.duration
    E = observed_durations.event_flag
    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=E)
    # kmf.plot()
    kmf.cumulative_density_.plot(figsize=(7,6))

    naf = NelsonAalenFitter()
    naf.fit(T,event_observed=E)
    plt.figure(figsize=(7,6))
    naf.plot()
    plt.title('Cumulative hazard rate')
    
    # =============================================================================
    #     Fit exponential cumulative hazard model
    #     https://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html
    # =============================================================================

    # wbf = WeibullFitter().fit(T, E, label='WeibullFitter')
    # pwf = PiecewiseExponentialFitter([40, 60]).fit(T, E, label='PiecewiseExponentialFitter')
    exf = ExponentialFitter().fit(T, E, label='ExponentialFitter')
    exf.plot_cumulative_hazard()
    print('fitted lambda = {}'.format(1 / exf.lambda_))
    # Confidence bounds on this?  --> bootstrap?
    
    # plot groundtruth curve
    plt.figure(figsize=(7,6))
    x = np.arange(1,30)
    plt.plot(x, expon(scale=1/target_rate).sf(x), 'g--', lw=2.5, alpha=.6, label='target')
    plt.plot(x, expon(scale=exf.lambda_).sf(x), 'r-',lw=3, alpha=.7, label='fitted')
    plt.legend()
    plt.xlabel('duration (time since event arrival')
    plt.title('Survival curve')
    
    # =============================================================================
    #     First go at a Bayesian estimate of lambda
    # =============================================================================
    lam_range = np.linspace(.01, .2, 390)
    prior = np.ones_like(lam_range)
    prior /= np.sum(prior)
    logprior = np.log(prior)
    logprior /= np.sum(logprior)
    
    # in original dimension (dangerously small numbers!)
    # post = prior
    # for _,row in observed_durations.iterrows():
    #     if row.event_flag==1:
    #         post *= expon(scale=1/lam_range).pdf(row.duration)
    #     else:
    #         post *= expon(scale=1/lam_range).sf(row.duration)     
    
    # in log dimension
    logpost = logprior #- lam_range*T.sum() + np.log(lam_range)*(1 - E).sum() # <-- Wrong
    for _,row in observed_durations.iterrows():
        if row.event_flag==1:
            logpost += expon(scale=1/lam_range).logpdf(row.duration)
        else:
            logpost += expon(scale=1/lam_range).logsf(row.duration)     
    
    post = np.exp(logpost)
    post /= np.sum(post)
    print('\nMean of lambda posterior = {}'.format(np.dot(lam_range, post)))
    
    plt.figure(figsize=(7,6))
    plt.plot(lam_range, post, 'b-', label='Bayes')
    plt.vlines(1 / exf.lambda_, 0, 1.2*np.max(post), color='m', lw=4, alpha=.6, label='MLE')
    plt.vlines(target_rate, 0, 1.2*np.max(post), color='g', lw=4, alpha=.6, label='target')
    plt.legend()
    plt.title('Lambda estimate')
    plt.xlabel('lambda')
   
    
    