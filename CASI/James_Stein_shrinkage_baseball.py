# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 21:04:27 2019

@author: Kris
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def D_binom(avg, N):
    """ factor D in JS formula,
        defined as (st.dev/n)**2
        ~'Normalised' Variance of binomial distribution
    """
    return avg * (1 - avg) / N

def James_Stein_factor(D, S, k):
    """ Equal variance version!
        This is the main factor in formula (2.10) from above paper, applied to 
        example in section 3.i, i.e. for binomial data.
        N refers to number of trials in binom model.
        k is number of samples in population.
    """
    return 1 - (k - 3)*D / S

# =============================================================================
# [1] James-Stein estimator for binomial observations with equal variance
# =============================================================================
# Batting averages example from paper "Biased versus Unbiased estimation"_ by Bradley Efron

batting = pd.read_excel('../../../data/Efron_baseball.xlsx')
batting['hits'] = batting['hits'].round(0).astype(int)
n_samples = len(batting) # =k
N = batting.loc[0,'at_bats']
global_average = batting.unbiased.mean()  # y stripe
S = np.sum((batting.unbiased.values - global_average)**2)
D = D_binom(global_average, N)
batting['JS_factor'] = James_Stein_factor(D, S, n_samples)
batting['JS_estimate2'] = batting.apply(lambda row: global_average + row.JS_factor * (row.unbiased - global_average), axis=1)  # vectorised version of formula (2.10)
batting['JS_estimate2'] = batting['JS_estimate2'].round(3)

# Compute number of standard deviations the JS estimate is away from original:
batting['n_stdevs'] = (batting['JS_estimate2'] - batting['unbiased']) / np.sqrt(D)

# Perform 'limited translation correction' by capping JS estimates to correction of max. 1 stdev
batting['JS_limited_translation'] = batting['JS_estimate2'].where(np.abs(batting['n_stdevs'])<=1, batting['unbiased'] + np.sign(batting['n_stdevs'])*np.sqrt(D))

# Plotting shrinkage effect  [[TO DO: add true averages to plot!]]
plt.figure(figsize=(10,4))
for _,row in batting[['unbiased','JS_limited_translation']].iterrows():
    plt.plot([row.unbiased, row.JS_limited_translation], [1, 0], 'bo-')
plt.plot([.1, .5], [1, 1], 'r--', alpha=.6, label='unbiased average')
plt.plot([.1, .5], [0, 0], 'g--', alpha=.6, label='James-Stein LT')
plt.ylim([-.1, 1.2])
plt.legend(fontsize=13)
plt.title("Equal variance James-Stein estimator: baseball example Efron(1975)",fontsize=16)

# =============================================================================
# [2] Add heteroscedasticity: unequal variances
# =============================================================================
# _Creating our own artificial dataset_
n_obs = 50
df = pd.DataFrame(data=np.random.exponential(scale=50, size=(n_obs,)), columns=['trials'])
df['trials'] = 1 + df['trials'].round(0).astype(int)  # +1 to prevent 0 trials
df['successes'] = np.random.uniform(0,1,n_obs) * df['trials']
df['successes'] = df['successes'].round(0).astype(int)
df['MLE'] = df['successes'] / df['trials']  # unbiased estimate

global_average = df['MLE'].mean()  # y stripe
S = np.sum((df.MLE.values - global_average)**2)
df['D'] = df['trials'].apply(lambda x: D_binom(global_average, x))  # Compute pooled variance for each observation!
df['sqrtD'] = np.sqrt(df['D'])
df['JS_factor'] = df['D'].apply(lambda D: James_Stein_factor(D, S, n_obs)) # individual multipliers
df['JS_factor'] = df['JS_factor'].where(df['JS_factor']>0, 0)  # enforce + condition

df['JS_estimate'] = df.apply(lambda row: global_average + row.JS_factor * (row.MLE - global_average), axis=1)  # vectorised version of formula (2.10)

# Perform 'limited translation correction' by capping JS estimates to correction of max. 1 stdev
df['n_stdevs'] = (df['JS_estimate'] - df['MLE']) / np.sqrt(df['D']) # number of standard deviations is the JS estimate away from MLE?
df['reduced_std'] = df['MLE'] + 1*np.sign(df['n_stdevs'])*df['sqrtD']
df['JS_estimate_LT'] = df['JS_estimate'].where(np.abs(df['n_stdevs'])<=1, df['reduced_std'])
print(df[abs(df.n_stdevs)>1])

# Plot result
plt.figure(figsize=(12,4))
for _,row in df[['MLE','JS_estimate_LT']].iterrows():
    plt.plot([row.MLE, row.JS_estimate_LT], [1, 0], 'bo-')
plt.plot([-.1, 1.1], [1, 1], 'r--', alpha=.6, label='unbiased MLE')
plt.plot([-.1, 1.1], [0, 0], 'g--', alpha=.6, label='James-Stein LT')
plt.ylim([-.1, 1.2])
plt.xlim([-.1, 1.3])
plt.legend(fontsize=13)
plt.title("Unequal variance James-Stein estimator: expon.- binom.",fontsize=16)

# Let's inspect largest differences
df_ = df.copy()
df_['delta_abs'] = np.abs(df['JS_estimate_LT'] - df['MLE'])
df_ = df_.sort_values(by='delta_abs', ascending=False)
print(df_[:10])

