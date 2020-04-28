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
#from scipy.integrate import quad
from scipy.stats import expon, lomax, gamma
from lifelines import KaplanMeierFitter
from lifelines.utils import datetimes_to_durations, median_survival_times
import matplotlib.pyplot as plt
from exponential_post_predictive import Bayesian_conjugate_inference
from decay_problem_MacKay import expon_integral

plt.style.use('bmh')


def add_durations(df, N=100, duration_min=1, lambd=5, groupname='A'):
    df['duration_days'] = [int(np.ceil(x)) for x in expon.rvs(size=N, loc=duration_min, scale=lambd)]
    df['group'] = groupname
    
def empty_df(N=100):
    DF = pd.DataFrame()
    dates = pd.date_range(start='2018-01-01', end='2019-01-01', freq='D')[:-1]
    DF['event0'] = [choice(dates) for _ in range(N)]  # replacement not supported by random.sample 
    DF['observation_date'] = pd.to_datetime('2019-01-01',format='%Y-%m-%d')
    return DF

def create_data(groups):
    df = pd.DataFrame()
    for name,params in groups.items():
        df_ = empty_df(params.size)
    
        # Assign durations (= 'time-to-event')
        add_durations(df_, params.size, params.min_delay, params.lambda_, name)
    
        # Assign non-conversions
        fail_rate = 1 - params.rate
        n_fails = int(fail_rate * params.size)
        df_.loc[list(sample(range(params.size), n_fails)), 'duration_days'] = np.nan

        # the rest
        df_['event1'] = df_.apply(lambda row: row.event0 + pd.to_timedelta(row.duration_days, unit='D'), axis=1)
        df_['event1'] = df_['event1'].where(df_.event1<=df_.observation_date, np.nan)
        df_['duration_days'] = df_['duration_days'].where(df_.event1<=df_.observation_date, np.nan)  # delays of events that are observed

        # Compute durations for K-M computation ('T' in lifelines.utils.datetimes_to_durations):
        #   -> For non-conversions: days relative to date of observation
        #   -> For conversions: observed days to conversion
        df_['duration_censored'] = (df_['observation_date'] - df_['event0']) / pd.to_timedelta(1, 'D')
        df_['duration_censored'] = df_['duration_days'].where(df_.duration_days>0, df_.duration_censored)

        # Add group to dataset
        df = pd.concat([df, df_], axis=1)
        
    df = df.sort_values(['group','event0'])
    df.index = ['E'+str(n).zfill(3) for n in range(len(df))]
    df.index.name = 'event'
    df['success'] = df['event1'].apply(lambda d: False if pd.isnull(d) else True) # 'E' in lifetimes
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
    lambda_range = np.linspace(.1, 50, 500)
    
    #     Generate test data
    # =============================================================================
    Groups = namedtuple('Groups', 'size min_delay lambda_ rate') # rate = rate of event taking place (~success)
    testgroups = {'A':Groups(size=100, min_delay=0, lambda_=20, rate=.7)}
    
    df = create_data(testgroups)
    df1 = df[['event0', 'duration_days', 'event1', 'duration_censored']]#[:40]
    print(df1.head(10))
    df.duration_days.hist()
    observed_delays = df['duration_days'].dropna().values # after right-truncation at 2019-01-01    
    
    print('\n%i/%i conversion events observed (rest censored).' % (df.success.sum(), len(df)))
    print('max. observable delay: %i days' % ((df.event0.max() - df.event0.min()) / pd.Timedelta(1,'D')))
    print('max. observed delay: %i days' % (df.duration_days.max()))
    print('\n')
    
    # True delay distribution
    True_delay = expon(loc=testgroups['A'].min_delay, scale=testgroups['A'].lambda_)
    
    # Compute observed delay dist. via Bayesian conj. priors (see script <exponential_post_predictive.py>)
    alpha_, beta_ = Bayesian_conjugate_inference(observed_delays - testgroups['A'].min_delay)
    # (marginal) posterior
    inverse_lambda_range = np.linspace(1/lambda_range[1], 1/lambda_range[0], len(lambda_range))
    lambd_dist_uncorrected = gamma(a=alpha_, loc=0, scale=1./float(beta_)).pdf(inverse_lambda_range)
    
    # Directly to posterior predictive
    ll = lomax(c=alpha_, loc=testgroups['A'].min_delay, scale=float(beta_))
    Observed_delay_dist_Bayesian = ll.pdf(x)

    # Compute 'corrected' delay dist. by adjusting likelihood function (see script <decay_problem_MacKay.py>)
#    df['days_ago'] = (today - df.event0) / pd.Timedelta(1,'D')  # ?!
#    completed_events = df[['delay_obs','days_ago']].dropna().copy()  # ?!
    #lambda_likelh = likelihood_over_all_observations(completed_events, min_delay=testgroups['A'].min_delay, lambda_range=lambda_range)
    
    lambda_likelh = pd.Series(data=np.ones((len(lambda_range),)))
    for obs in observed_delays:
        Z = expon_integral(lambda_range, 0, obs) # integration constants, dependent on observation window but independent of observation        
        likelihood_of_lambda_for_this_observation = expon.pdf(obs, loc=testgroups['A'].min_delay, scale=lambda_range) / Z
        lambda_likelh *= likelihood_of_lambda_for_this_observation
    #lambda_likelh /= np.sum(lambda_likelh.values)
    #lambda_expected_value = (lambda_likelh.values*lambda_range).sum()

    # Compute proper posterior predictive from lambda likelihood, as mixture pmf [to do: incorporate better lambda prior]
    # construct simple uniform prior on lambda with arbitrary truncations (correcting wild )
    lambda_prior = np.zeros_like(lambda_range)
    lambda_prior[(lambda_range>=5) & (lambda_range<=60)] = 1  # adjust the posts
    lambda_prior /= np.sum(lambda_prior)
    lambda_post = lambda_prior * lambda_likelh
    lambda_post /= np.sum(lambda_post)
    lambda_expected_value = np.dot(lambda_post, lambda_range)
    
#    # put together predictive posterior
#    Mix_pmf = np.zeros((len(x), ), dtype=float)  # Recall that x is range of delays
#    for i,delay in enumerate(x):
#        for lambd, Prob in zip(lambda_range, lambda_range):
#            exponential = expon.pdf(delay, loc=testgroups['A'].min_delay, scale=lambd)
#            Mix_pmf[i] += exponential * Prob
#    Mix_pmf /= np.sum(Mix_pmf)
    
    # Plot lambda posterior
    # ==========================
    plt.figure(figsize=(9,6))
    plt.plot(lambda_range, lambda_prior, 'k--', lw=1, alpha=.7, label='prior')
    plt.plot(lambda_range, lambda_post, 'm-', lw=1.5, label='corrected')
    #plt.plot(lambda_range, lambd_dist_uncorrected, 'm:', lw=2, label='uncorrected')
    plt.vlines(lambda_expected_value, 0, np.max(lambda_post), color='m', linewidth=2, label='Uncorrected')
    #plt.vlines(lambda_expected_value, 0, np.max(lambda_likelh), color='b', linewidth=3, label='mean estimated')
    plt.vlines(testgroups['A'].lambda_, 0, np.max(lambda_post), color='g', linewidth=5, alpha=.6, label='True')
    
    plt.xlabel('lambda', fontsize=14)
    plt.ylabel('density', fontsize=14)
    plt.legend(fontsize=15, loc=2)
    plt.xlim(0,30)
    plt.title('Likelihood of lambda P({X}|lambda)', fontsize=14)
    
    #     Plot posterior predictive of actual delays
    # =============================================================================
#    plt.figure(figsize=(9,6))
#    df.delay_true.hist(bins=int(df.delay_true.max()/2), alpha=.5, density=True, color='y', label='true')
#    df.delay_obs.hist(bins=int(df.delay_obs.max()/2), alpha=.2, density=True, color='b', label='observed')
#    plt.legend(fontsize=14)
#    plt.title('Observed vs unobserved delays', fontsize=15)
    
    plt.figure(figsize=(9,6))
    True_delay_norm = True_delay.pdf(x) / np.sum(True_delay.pdf(x))
    plt.plot(x, True_delay_norm, 'g-', lw=4, alpha=.5, label='True')
    #plt.plot(x, expon.pdf(x, loc=2, scale=1/g.mean()), 'y-', label='uncorrected, expected lambda')
    Observed_delay_dist_Bayesian /= np.sum(Observed_delay_dist_Bayesian)
    plt.plot(x, Observed_delay_dist_Bayesian, 'y--', lw=2.5, label='uncorrected')    
    corrected_mean_lambda = expon.pdf(x, loc=testgroups['A'].min_delay, scale=lambda_expected_value)
    corrected_mean_lambda /= np.sum(corrected_mean_lambda)
    plt.plot(x, corrected_mean_lambda, 'm-.', lw=2.5, alpha=.6, label='corrected (mean lambda)')
    #plt.plot(x, Mix_pmf, 'm:', lw=2.5, alpha=.6, label='corrected (mixture)')
    plt.xlim(0,40)
    plt.legend(fontsize=14)
    plt.title('Delay distribution', fontsize=15)
    
    plt.figure(figsize=(6,4))
    Means_compare = pd.DataFrame(index=['True','Uncorrected','Corrected'], 
                                 columns=['Expected Value'], \
                                 data=[np.dot(x,True_delay_norm), \
                                       np.dot(x,Observed_delay_dist_Bayesian), \
                                       np.dot(x,corrected_mean_lambda)])
    Means_compare.plot.bar(width=1)
    
    # KM estimate    
    kmf = KaplanMeierFitter()
    T, E = df.duration_censored.values, df.success.values
    kmf.fit(T, event_observed=E) 
    #kmf.cumulative_density_
    kmf.plot()
    median_ = kmf.median_survival_time_
    print(median_)
    print(np.median(observed_delays))
    
# =============================================================================
#     TO DOs:
#         - ADD K-M mean estimate
#         - Fix posterior plot
#         - ADD MLE-fitted parametric exponential model
#         - Monte Carlo: repeat 500 times and collect variation
#         
# =============================================================================
