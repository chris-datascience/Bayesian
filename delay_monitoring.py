# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 21:21:26 2019

@author: Kris
"""

# =============================================================================
# Monitor exponential delays in one-event survival data over time
# =============================================================================

import os, sys
from random import sample, choice
from collections import namedtuple
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import expon, lomax, gamma
import matplotlib.pyplot as plt
from exponential_post_predictive import Bayesian_conjugate_inference
from decay_problem_MacKay import expon_integral

plt.style.use('bmh')


def add_durations(df, N=100, duration_min=1, lambd=5, groupname='A'):
    #durations = np.array(int(mu*N) + (sigma*N)*np.random.randn(N,1), dtype=int)  # time to event durations normally distributed
    #durations = expon(loc=1, scale=lambd)
    durations = [int(x) for x in expon.rvs(size=N, loc=duration_min, scale=lambd)]
    df['days_to_conversion'] = np.clip(durations, 0, int(N))
    df['group'] = groupname
    
def empty_df(N=100):
    DF = pd.DataFrame()
    dates = pd.date_range(start='2018-01-01', end='2019-01-01', freq='D')
    DF['first_contact'] = [choice(dates) for _ in range(N)]  # replacement not supported by random.sample 
    return DF

def create_data(groups, endtime=True):
    df = pd.DataFrame()
    for name,params in groups.items():
        df_ = empty_df(params.size)
    
        # Assign durations (= 'time-to-event')
        add_durations(df_, params.size, params.min_delay, params.lambda_, name)
    
        # Assign non-conversions
        fail_rate = 1 - params.rate
        n_fails = int(fail_rate * params.size)
        df_.loc[list(sample(range(params.size), n_fails)), 'days_to_conversion'] = np.nan
    
        # Add group to dataset
        df = pd.concat([df, df_])
        
    df = df.sort_values('first_contact')
    df.index = ['L'+str(n).zfill(3) for n in range(len(df))]
    df.index.name = 'lead'
    df['conversion_date'] = df.apply(lambda row: row.first_contact + pd.to_timedelta(row.days_to_conversion, unit='D'), axis=1)
    df['success'] = df['conversion_date'].apply(lambda d: False if pd.isnull(d) else True)
    
    if endtime:
        # Compute durations for K-M computation ('E' in lifelines.utils.datetimes_to_durations):
        #   -> For non-conversions: days relative to date of observation
        #   -> For conversions: observed days to conversion
        observation_date = pd.to_datetime('2019-02-01', format='%Y-%m-%d')
        df['end_time'] = (observation_date - df['first_contact']) / pd.to_timedelta(1,'D')
        df['end_time'] = df['end_time'].where(np.isnan(df.days_to_conversion), df.days_to_conversion)
    return df

def probability_of_observation(x, obs_window=(1,20)):
    x_obs_min, x_obs_max = obs_window
    if all([x>=x_obs_min, x<=x_obs_max]):
        return 1
    else:
        return 0

def likelihood_over_all_observations(observations, min_delay=0, lambda_range=np.linspace(.1, 20, 500)):
    """  For exponential distribution only!
         see <decay_problem_MacKay.py>
         'observations' is dataframe with
    """
    final_likelihood_lam = pd.DataFrame(data=lambda_range, columns=['all_obs'])
    final_likelihood_lam['all x'] = 1
    for _,obs in observations[5:6].iterrows():
        Z = expon_integral(lambda_range, 0, obs.days_ago - min_delay) # integration constants, dependent on observation window but independent of observation        
        print(pd.isnull(Z).sum())
        
        likelihood_of_lambda_for_this_observation = expon.pdf(obs.delay_obs, loc=min_delay, scale=lambda_range) / Z
        
        final_likelihood_lam['all_obs'] *= likelihood_of_lambda_for_this_observation
    final_likelihood_lam['all_obs'] /= np.sum(final_likelihood_lam['all_obs'])
    return final_likelihood_lam['all_obs']


if __name__=='__main__':
    
    x = np.linspace(0,50,401) # evaluation points of delay functions
    lambda_range = np.linspace(.5, 20, 100)
    
    #     Generate test data
    # =============================================================================
    Groups = namedtuple('Groups', 'size min_delay lambda_ rate') # rate = rate of event taking place (~success)
    testgroups = {'A':Groups(size=300, min_delay=2, lambda_=10, rate=.6)}
    
    df = create_data(testgroups, endtime=False)
    df = df[['first_contact', 'days_to_conversion', 'conversion_date']][:40]
    df.columns = ['event0','delay_true','event1_true']
    df['delay_obs'] = df['delay_true'].where(df.event1_true<=df.event0.values[-1], np.nan)  # delays of events that are observed
    observed_delays = df['delay_obs'].dropna().values
    df['event1_obs'] = df['event1_true'].where(df.event1_true<=df.event0.values[-1], np.nan) # observed at time of final date of event0 column
    print('\n%i/%i conversion events not observed.' % (df['event1_obs'].isnull().sum() - df['event1_true'].isnull().sum(), len(df.delay_true.dropna())))
    print('max. observable delay: %i days' % ((df.event0.max() - df.event0.min()) / pd.Timedelta(1,'D')))
    print('max. observed delay: %i days' % (df.delay_obs.max()))
    print('\n')
    
    # True delay distribution
    True_delay = expon(loc=testgroups['A'].min_delay, scale=testgroups['A'].lambda_)
    
    # Compute observed delay dist. via Bayesian conj. priors (see script <exponential_post_predictive.py>)
    alpha_, beta_ = Bayesian_conjugate_inference(observed_delays - testgroups['A'].min_delay)
    # (marginal) posterior
    lambd_dist_uncorrected = gamma(alpha_, loc=0, scale=1./float(beta_)).pdf(lambda_range)
    
    # posterior predictive
    ll = lomax(c=alpha_, loc=testgroups['A'].min_delay, scale=float(beta_))
    Observed_delay_dist_Bayesian = ll.pdf(x)

    # Compute corrected delay dist. by adjusting likelihood function (see script <decay_problem_MacKay.py>)
    today = df.event0.values[-1]
    df['days_ago'] = (today - df.event0) / pd.Timedelta(1,'D')
    
    completed_events = df[['delay_obs','days_ago']].dropna().copy()
    #lambda_likelh = likelihood_over_all_observations(completed_events, min_delay=testgroups['A'].min_delay, lambda_range=lambda_range)
    
    lambda_likelh = pd.Series(data=np.ones((len(lambda_range),)))
    for _,obs in completed_events.iterrows():
        Z = expon_integral(lambda_range, 0, obs.days_ago) # integration constants, dependent on observation window but independent of observation        
        likelihood_of_lambda_for_this_observation = expon.pdf(obs.delay_obs, loc=2, scale=lambda_range) / Z
        lambda_likelh *= likelihood_of_lambda_for_this_observation
    lambda_likelh /= np.sum(lambda_likelh.values)
    lambda_expected_value = (lambda_likelh.values*lambda_range).sum()

    # Compute proper posterior predictive from lambda likelihood, as mixture pmf [to do: incorporate better lambda prior]
    # =============================================================================
    # construct simple uniform prior on lambda with arbitrary truncations (correcting wild )
    lambda_prior = np.zeros_like(lambda_range)
    lambda_prior[(lambda_range>.75) & (lambda_range<17)] = 1
    lambda_post = lambda_prior * lambda_likelh
    # put together predictive posterior
    Mix_pmf = np.zeros((len(x), ), dtype=float)  # Recall that x is range of delays
    for i,delay in enumerate(x):
        for lambd, Prob in zip(lambda_range, lambda_range):
            exponential = expon.pdf(delay, loc=testgroups['A'].min_delay, scale=lambd)
            Mix_pmf[i] += exponential * Prob
    Mix_pmf /= np.sum(Mix_pmf)
    
    # Plot lambda posterior
    plt.figure(figsize=(9,6))
    plt.plot(lambda_range, lambda_prior*np.mean(lambda_likelh), 'k--', lw=1, label='prior')
    plt.plot(lambda_range, lambda_likelh, 'k-', lw=2.5, label='corrected')
    plt.plot(lambda_range, lambd_dist_uncorrected, 'm:', lw=2, label='uncorrected')
    plt.vlines(lambda_expected_value, 0, np.max(lambda_likelh), color='b', linewidth=3, label='mean estimated')
    plt.vlines(testgroups['A'].lambda_, 0, np.max(lambda_likelh), color='g', linewidth=3, label='True')
    plt.xlabel('lambda', fontsize=14)
    plt.ylabel('density', fontsize=14)
    plt.legend(fontsize=15, loc=2)
    plt.xlim(0,20)
    plt.title('Likelihood of lambda P({X}|lambda)', fontsize=14)
    
    #     Plot posterior predictive of actual delays
    # =============================================================================
    plt.figure(figsize=(9,6))
    df.delay_true.hist(bins=int(df.delay_true.max()/2), alpha=.5, density=True, color='y', label='true')
    df.delay_obs.hist(bins=int(df.delay_obs.max()/2), alpha=.2, density=True, color='b', label='observed')
    plt.legend(fontsize=14)
    plt.title('Observed vs unobserved delays', fontsize=15)
    
    plt.figure(figsize=(9,6))
    True_delay_norm = True_delay.pdf(x) / np.sum(True_delay.pdf(x))
    plt.plot(x, True_delay_norm, 'g-', lw=4, alpha=.5, label='True')
    #plt.plot(x, expon.pdf(x, loc=2, scale=1/g.mean()), 'y-', label='uncorrected, expected lambda')
    Observed_delay_dist_Bayesian_norm = Observed_delay_dist_Bayesian/np.sum(Observed_delay_dist_Bayesian)
    plt.plot(x, Observed_delay_dist_Bayesian_norm, 'y-', lw=2.5, label='uncorrected')
    
    corrected_mean_lambda = expon.pdf(x, loc=2, scale=lambda_expected_value)
    corrected_mean_lambda_norm = corrected_mean_lambda / np.sum(corrected_mean_lambda)
    plt.plot(x, corrected_mean_lambda_norm, 'm--', lw=2.5, alpha=.6, label='corrected (mean lambda)')
    plt.plot(x, Mix_pmf, 'b--', lw=2.5, alpha=.6, label='corrected (mixture)')
    plt.xlim(0,40)
    plt.legend(fontsize=14)
    plt.title('Delay distribution', fontsize=15)
    
    print(df.tail(15))    
    
