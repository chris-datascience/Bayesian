# -*- coding: utf-8 -*-
"""
Created on Tue May 26 21:38:32 2020

@author: erdbrca
"""


# =============================================================================
# This explores ways to artificially generate survival datasets with desired properties
# =============================================================================

import os, sys
import numpy as np
import pandas as pd
from scipy.stats import expon, binom
# from lifelines import KaplanMeierFitter
# from lifelines import NelsonAalenFitter

import matplotlib.pyplot as plt
plt.style.use('bmh')


def generate_constant_hazard(n_events=100, t_max=40, target_rate=.1, noise_factor=.05):
    """ Generate dataset with constant hazard rate (independent of time)
        i.e. h(t) = constant
        i.e. exponential survival function.
        
        NB. Noise currently not implemented. Rate is scalar (previously vector of len=t_max)
    """
    
    #rate = target_rate * np.ones((t_max,)) + noise_factor * np.random.randn(t_max)  # as function of discrete time variable!
    rate = target_rate
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
    # Add extra/remaining observations
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
    return observed_durations
