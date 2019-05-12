# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:36:11 2019

@author: erdbrca
"""

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy.stats import dirichlet as dlt

"""
    Refs:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet.html
        https://en.wikipedia.org/wiki/Categorical_distribution
"""

# Settings
plot_priors = True

# Observed Data
count_obs = OrderedDict({'id1':87, 'id2':34, 'id3':1})
counts = np.array(list(count_obs.values()), dtype=int)

dirichlet_prior = np.ones_like(counts)  # uninformative prior
dirichlet_posterior = dirichlet_prior + counts
prior_samples = np.random.dirichlet(dirichlet_prior, size=100000)
posterior_samples = np.random.dirichlet(dirichlet_posterior, size=100000)

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
    plt.legend(fontsize=15)
    plt.title('Prior Probs', fontsize=16)
    plt.xlabel('P (-)', fontsize=15)

# Plot posteriors
plt.figure(figsize=(9,6))
for i, label in enumerate(count_obs.keys()):
    ax = plt.hist(posterior_samples[:,i], density=True, bins=50, label=label, alpha=.75, histtype='stepfilled')
    ax = plt.plot([PoM[i] + 1.65*np.sqrt(PoV[i])]*2, [0,20], 'k--', lw=3)  # plot decision boundaries
plt.legend(fontsize=15)
plt.title('posterior Probs', fontsize=16)
plt.xlabel('P (-)', fontsize=15)

# Replicate prior pdfs
n = 3
T = np.tile(np.linspace(.01,.99,5), (n,1))
print('\n..FAILED Attempt at getting multiple pdf points per variable:', dlt.pdf(T/np.sum(T,0), dirichlet_prior))

