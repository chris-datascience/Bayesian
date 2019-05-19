# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:36:11 2019

@author: erdbrca
"""

import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy.stats import dirichlet as dlt

"""
    Applies to multinomial, categorical distributions with Dirichlet priors.
    Refs:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet.html
"""

def get_samples(alpha_vector, N=10000, method='scipy'):
    if method=='scipy':
        return dlt.rvs(alpha_vector, size=N)
    elif method=='numpy':
        return np.random.dirichlet(alpha_vector, size=N)

def compute_LPV_from_samples(alpha_vector):
    post_samples = get_samples(alpha_vector)    
    """
        LPV = least plausible vector, see BMH sec.4.3.4
    """
    Ncats = post_samples.shape[1]
    LPV = np.empty((Ncats,))
    for c in range(Ncats):
        LPV[c] = np.percentile(post_samples[:,c], 20)  # to do: vectorise
    return LPV

def compute_LPV_from_parameters(alpha_vector):
    M = dlt.mean(alpha_vector)
    V = dlt.var(alpha_vector)
    LPV = M - 1.65*np.sqrt(V)  # 5-percentile
    return np.where(LPV<0, 0, LPV)

def compute_rating(LPV_vector, method='max'):
    if method=='max':
        return np.max(LPV_vector)
    elif method=='top2':
        # score based on sum of top 2
        return np.sum(np.sort(LPV_vector)[-2:])
    elif method=='flat':
        return np.sum(np.abs(LPV_vector - 1/LPV_vector.shape[0]))
        
# =============================================================================
# Example 1: one set of observations, choice between id1, id2, id3
# =============================================================================
# Settings
plot_priors = False
n_samples = 10000

# Observed Data
count_obs = OrderedDict({'id1':87, 'id2':34, 'id3':1})
counts = np.array(list(count_obs.values()), dtype=int)

dirichlet_prior = np.ones_like(counts)  # uninformative prior based on pseudo-counts
dirichlet_posterior = dirichlet_prior + counts
prior_samples = get_samples(dirichlet_prior)
posterior_samples = get_samples(dirichlet_posterior)

print('prior means: %s' % (str(dlt.mean(dirichlet_prior))))
PoM = dlt.mean(dirichlet_posterior)
print('posterior means: %s' % (str(PoM)))
PoV = dlt.var(dirichlet_posterior)
print('posterior variances: %s' % (str(PoV)))
print('naive posterior means: %s' % ((counts + 1) / np.sum(counts + 1))) # expected from value counts plus assumed prior counts
print('Entropy DLT prior:', dlt.entropy(dirichlet_prior))
print('Entropy DLT posterior:', dlt.entropy(dirichlet_posterior))

if plot_priors:
    plt.figure(figsize=(9,6))
    for i, label in enumerate(count_obs.keys()):
        ax = plt.hist(prior_samples[:,i], bins=50, density=True, alpha=.35, label=label, histtype='stepfilled')
        print('sampled', i, ':  ', np.mean(prior_samples[:,i]))
        #if i==0: plt.plot(np.linspace(0,1,1000), DLT_[:,1], 'k-', alpha=.7, label=label)
    plt.legend(fontsize=15)
    plt.title('Prior Probs', fontsize=16)
    plt.xlabel('P (-)', fontsize=15)

# Plot posteriors
plt.figure(figsize=(8,5))
for i, label in enumerate(count_obs.keys()):
    percentile95 = PoM[i] + 1.65*np.sqrt(PoV[i])
    ax = plt.hist(posterior_samples[:,i], density=True, bins=50, label=label, alpha=.5, histtype='stepfilled')
    ax = plt.plot([percentile95]*2, [0,20], 'k--', lw=2)  # plot decision boundaries
plt.legend(fontsize=15)
plt.title('posterior Probs', fontsize=16)
plt.xlabel('P (-)', fontsize=15)

# =============================================================================
# Example 2: Comparing observations from different populations
# 'obs' = observation, consisting of counts in 'n_cat' categories
# =============================================================================
# Create dataset
n_cat, n_obs = 3, 10
Obs = pd.DataFrame(data=np.random.randint(low=1, high=25, size=(n_obs, n_cat)), \
                   columns=list('ABC'),
                   index=['obs_'+str(i) for i in range(n_obs)], \
                   dtype=int)
Obs.plot.bar(figsize=(7,4), stacked=True)
plt.ylabel('total count')
plt.title('Unordered observations')

# Perform inference
Alpha_post = (Obs.copy() + np.ones(Obs.shape)).astype(int)
LPV_scores_param = Alpha_post.apply(lambda row: compute_LPV_from_parameters(row.values))
LPV_scores_sampled = Alpha_post.apply(lambda row: compute_LPV_from_samples(row.values))
Alpha_post['ratings'] = LPV_scores_sampled.apply(lambda row: compute_rating(row.values, method='flat'), axis=1)
Alpha_post = Alpha_post.sort_values(by='ratings', ascending=True)  # False if using other methods
Alpha_post[Obs.columns].plot.bar(figsize=(7,4), stacked=True)
plt.ylabel('total count')
plt.title('Ranked acc. to evenness of distribution over categories')