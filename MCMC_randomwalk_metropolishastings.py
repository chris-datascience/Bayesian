# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:40:35 2020

@author: Kris
"""

import time
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Coursera Spring 2020 <<Bayesian Statistics: Techniques and models>>
#
# Lesson 4.3 Random-Walk Metropolis-Hastings example
# =============================================================================

def lg(mu, n, ybar):
    mu2 = mu**2
    return n * (ybar * mu - mu2 / 2.) - np.log(1 + mu2)

def mh(n, ybar, n_iter=1000, mu_init=0, cand_sd=2):
  ## Random-Walk Metropolis-Hastings algorithm
  
  ## step 1, initialize
  mu_out = np.empty(n_iter)
  accpt = 0
  mu_now = mu_init
  lg_now = lg(mu_now, n, ybar)
  uniform_samples = np.random.rand(n_iter)
  
  ## step 2, iterate
  for i in range(n_iter):
    ## step 2a
    mu_cand = np.random.normal(loc=mu_now, scale=cand_sd, size=1) # draw a candidate
    
    ## step 2b
    lg_cand = lg(mu_cand, n, ybar) # evaluate log of g with the candidate
    lalpha = lg_cand - lg_now # log of acceptance ratio
    alpha = np.exp(lalpha)
    
    ## step 2c
    # Draw a uniform variable which will be less than alpha with probability min(1, alpha)
    if uniform_samples[i] < alpha:
        # then accept the candidate
        mu_now = mu_cand
        accpt += 1 # to keep track of acceptance
        lg_now = lg_cand
    
    ## collect results
    mu_out[i] = mu_now # save this iteration's value of mu
  
  return mu_out, accpt / n_iter


np.random.seed(13)
N_samples = 10000

y = np.array([1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3, 1.9], dtype=float)
ybar = np.mean(y)
n = len(y)
plt.hist(y, density=True)
plt.title('Samples (Observations)', fontsize=15)

t0 = time.time()
mu_post, accept_rate = mh(n, ybar, n_iter=N_samples, mu_init=0., cand_sd=1.5)
print('Stopwatch 1:   %2.3f sec' % (time.time() - t0))
print('Acceptance rate: %.2f' % accept_rate)
plt.figure(figsize=(10,5))
plt.plot(mu_post, 'b-')
plt.title('trace',fontsize=15)

plt.figure(figsize=(7,5))
plt.hist(mu_post[-4000:], bins=31, color='DarkGreen', density=True)
plt.xlabel('mu')
plt.title('posterior dens. mu', fontsize=15)
