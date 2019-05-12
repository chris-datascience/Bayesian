# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 12:38:45 2019

@author: Kris
"""


"""
    Modelling relative volume growth using Gibbs Sampling: 
    1. Bayesian inference on normal distribution with unknown mean and st.dev
    mu is Normal(mu, s2), stdev is InverseGamme(a,b)
    Source:
        https://www4.stat.ncsu.edu/~reich/ABA/code/NN2
        [translated from R]
        which appears on
        https://www4.stat.ncsu.edu/~reich/ABA/code.html

    Other Inspirational sources:
        https://docs.pymc.io/notebooks/updating_priors.html
        https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter4_TheGreatestTheoremNeverTold/Ch4_LawOfLargeNumbers_PyMC3.ipynb

    Zeros in input volumes series should not contain zeros.
"""

import numpy as np
import matplotlib.pyplot as plt

# Create dummy dataset
#price_observations = np.array([1.8, 2.9, 3.3, 4.0, 4.9], dtype=np.float)
#observed_returns = np.diff(price_observations) / price_observations[:-1]  # pct_change
#Y = observed_returns.copy()

true_mu = 1.0
true_sigma = .2
N_obs = 6
Y = true_mu + true_sigma * np.random.randn(N_obs,)

# Priors
mu0 = 0.
stdev0 = 1000.
a = .01
b = .01

N_samples = 40000
n = len(Y)
mus = np.empty(N_samples, dtype=float)  # sampled mu
stdev2s = np.empty(N_samples, dtype=float)  #  sampled variance

# Initialisation
mu = np.mean(Y)
var_ = np.var(Y)

# Sample from conditional posteriors
for i in range(1, N_samples):
    # sample mu|var_,Y
    A1 = np.sum(Y) / var_ + mu0 / stdev0
    B1 = n / var_ + 1. / stdev0
    mu = np.random.normal(loc=A1/B1, scale=1./np.sqrt(B1), size=1)
    
    # sample var_|mu,Y
    A2 = n / 2. + a
    B2 = np.sum((Y - mu)**2) / 2 + b
    var_ = 1. / np.random.gamma(A2, scale=1./B2, size=1)  #  cf. syntax of R-command 'rgamma'
    
    # store variables
    mus[i] = mu
    stdev2s[i] = var_

# Cutting off burn-in period
mus_good = mus[20000:]
stdev2s_good = stdev2s[20000:]
stdevs_good = np.sqrt(stdev2s_good)

# POST
print('Summary:')
print('mu mean: %1.3f' % np.mean(mus_good))
print('mu std: %1.3f' % np.std(mus_good))
print('stdev mean: %1.3f' % np.mean(stdevs_good))
print('stdev std: %1.3f' % np.std(stdevs_good))

# Trace plots after burn-in    
plt.figure(figsize=(12,4))
plt.plot(mus_good, 'b-', alpha=.2, label='mu')
plt.title('Trace plot mu')
plt.figure(figsize=(12,4))
plt.plot(stdevs_good, 'g-', alpha=.2, label='stdev')
plt.plot('Trace plot stdev')

plt.figure(figsize=(7,6))
plt.hist(mus_good, bins=75, color='b', density=True, alpha=.7, label='posterior mu')
plt.vlines(np.mean(mus_good), 0, 1., color='r', label='expected value')
plt.vlines(true_mu, 0, 1., color='k', label='true mu')
plt.legend()

plt.figure(figsize=(7,6))
plt.hist(stdevs_good, bins=75, color='g', density=True, alpha=.7, label='posterior stdev')
plt.vlines(np.mean(stdevs_good), 0, 1., color='r', label='expected value')
plt.vlines(true_sigma, 0, 1., color='k', label='true mu')
plt.legend()

# Now supposing the input series represent relative growth of a merchant's volume, or stock returns, then:
Z_factor = 1.65
plt.figure(figsize=(9,7))
plt.plot(Y, 'ko-', label='returns')
plt.fill_between(range(len(Y)+1), np.mean(mus_good) + Z_factor*np.std(mus_good), np.mean(mus_good) - Z_factor*np.std(mus_good), color='b', alpha=.3)
plt.hlines(np.mean(mus_good), 0, len(Y), color='b', lw=4, alpha=.7, label='Gibbs mean')
plt.legend(fontsize=15)



