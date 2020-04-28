# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:19:01 2020

@author: Kris
"""

import numpy as np
import pandas as pd
import time
#from scipy.optimize import minimize
from scipy.stats import gamma as gamma_dist
from scipy.special import gamma
from scipy.stats import poisson, norm
import matplotlib.pyplot as plt
from math import factorial
import seaborn as sns
sns.set_style(style='dark')

"""
    Constructing a posterior predictive distribution by combining finite mixtures, 
      i.e. corresponding to individual parameter posteriors.
    In other words, 'combining' here means marginalising by grid approximation.
    See https://en.wikipedia.org/wiki/Compound_probability_distribution
    Interesting to compare to Monte Carlo methods, e.g. collapsed Gibbs sampling.
"""


def Pois(lambd, k):
    return (np.exp(-lambd) * lambd**k) / factorial(k)

def Gamma_dist(alpha, beta, x):
    return (beta**alpha) * (x**(alpha - 1)) * np.exp(-beta * x) / gamma(alpha)
 


if __name__=='__main__':
    # In 1D, i.e. for a single parameter distribution.
    # taking Poisson dist. as an example.
    lambda_mean = 4
    lambda_scale = 1.5
    n_posterior_samples = 1000
    rv = norm(loc=lambda_mean, scale=lambda_scale)
    lambda_range = np.linspace(0, 15, n_posterior_samples)
    lambda_post = rv.pdf(lambda_range)
    plt.plot(lambda_range, lambda_post, 'b-')
    plt.xlabel('lambda')
    plt.title('parameter lambda posterior')
    
    # [Method 1: lists of tuples & pd.Series]
    post_pred_val_range = np.arange(0,21)  # target values of final dist.
    PostPred_single_lambda = pd.Series(index=post_pred_val_range, \
                                       data=[Pois(lambda_mean, x) for x in post_pred_val_range]) # for reference
    PostPred_single_lambda.name = 'single'
    PostPred = pd.Series(index=post_pred_val_range, data=0)
    PostPred.name = 'marginalised'
    lambda_pmf = zip(lambda_range, lambda_post)
    for lambda_val, lambda_prob in lambda_pmf:
        for k in post_pred_val_range:
            PostPred.loc[k] += lambda_prob * Pois(lambda_val, k)
    PostPred /= PostPred.sum()
    
    Post = pd.concat([PostPred, PostPred_single_lambda], axis=1)
    Post.plot(style=['s-', 's-'], alpha=.7, figsize=(9,6), fontsize=12)
    plt.legend(fontsize=13)    
    plt.xlabel('k', fontsize=13)
    plt.title('Posterior Predictive distributions', fontsize=14)

    # [Method 2: Same example, now with Numpy]
    parameter_vector = np.array(lambda_post, dtype=float)
    PostPred_matrix = np.empty((len(post_pred_val_range), len(lambda_range)))
    for i,lambd in enumerate(lambda_range):
        PostPred_matrix[:,i] = np.array([Pois(lambd,k) for k in post_pred_val_range], dtype=float)  # Assuming PMF cannot be called in vectorised way in this case
        PostPred_matrix[:,i] /= PostPred_matrix[:,i].sum()
    PostPred_dist = PostPred_matrix.dot(parameter_vector)
    PostPred_dist /= np.sum(PostPred_dist)
    Post['numpy_mix'] = PostPred_dist
    Post.plot(style=['s-', 's-', 'o--'], alpha=.7, figsize=(9,6), fontsize=12)
    
    # [Method 3: extension to multiple parameters & broadcasting PDF calls]
    # ---------------------------------------------------------------------
    n_alpha_samples, n_beta_samples = 25, 45
    n_gamma_pts = 60
    post_pred_val_range = np.linspace(.1, 10, n_gamma_pts)  # pts to eval gamma vals
    alpha0, beta0 = 5, 2
    #G = Gamma_dist(alpha0, beta0, post_pred_val_range) # single parameter posterior predictive
    G = gamma_dist.pdf(post_pred_val_range, a=alpha0, loc=0, scale=1./beta0)
    plt.figure()
    plt.plot(post_pred_val_range, G, 'k-')
    # parameter priors:
    rv1 = norm(loc=alpha0, scale=1)
    rv2 = norm(loc=beta0, scale=.3)
    alpha_range = np.linspace(.5, 10, n_alpha_samples)
    beta_range = np.linspace(.5, 4, n_beta_samples)
    alpha_post = rv1.pdf(alpha_range)
    alpha_post /= np.sum(alpha_post)
    beta_post = rv2.pdf(beta_range)
    beta_post /= np.sum(beta_post)
    plt.figure()
    plt.plot(alpha_range, alpha_post, 'b-', label='alpha')
    plt.plot(beta_range, beta_post, 'r-', label='beta')
    plt.legend()
    plt.title('Parameter posteriors')
    
    # MAIN: BROADCASTING
    # TO DO: generalise for multiple-level hierarchy, using reduce() ?
    # ================================================================
    def compute_post_mix(g_val, Aprob, Bprob, Aval, Bval):
        """ TO DO: make flexible by using *args """
        return Aprob * Bprob * gamma_dist.pdf(g_val, Aval, loc=0, scale=1/Bval) # can use np.multiply or np.einsum for speed?!

    # TAKE 1
    t0 = time.time()
    A_values, B_values, G_values = np.meshgrid(alpha_range, beta_range, post_pred_val_range, indexing='ij')
    A_probs, B_probs, _ = np.meshgrid(alpha_post, beta_post, post_pred_val_range, indexing='ij')
    PostPredictive_raw = compute_post_mix(G_values, A_probs, B_probs, A_values, B_values)
    PostPredictive = PostPredictive_raw.sum(axis=(0,1))
    PostPredictive /= PostPredictive.sum()
    print('Broadcasting with meshgrid took {} sec.'.format(np.round(time.time() - t0, 2)))
    #plt.plot(post_pred_val_range, PostPredictive)

    # TAKE 2
    t0 = time.time()
    #PostPredictive_2 = compute_post_mix(post_pred_val_range.reshape(1,-1), alpha_post.reshape(-1,1), beta_post.reshape(1,-1), alpha_range.reshape(-1,1), beta_range.reshape(1,-1))
    #PostPredictive_2 = PostPredictive_2.sum(axis=(0,1))
    PostPredictive_2 = np.einsum('ijk,ijk,ijk->k', A_probs, B_probs, gamma_dist.pdf(G_values, A_values, loc=0, scale=1/B_values))
    PostPredictive_2 /= PostPredictive_2.sum()
    print(np.allclose(PostPredictive, PostPredictive_2))  # check it matches up with above
    print('Broadcasting with meshgrid + einsum took {} sec.'.format(np.round(time.time() - t0, 2)))

    # Re-computing with loops to validate above
    t0 = time.time()
    PostPred_dist = pd.DataFrame(index=range(len(post_pred_val_range)), data=np.zeros((n_gamma_pts,)), columns=['mixture'])
    for alpha, alpha_prob in zip(alpha_range, alpha_post):
        for beta, beta_prob in zip(beta_range, beta_post):
            for i,gamma_val in enumerate(post_pred_val_range):
                PostPred_dist.loc[i,'mixture'] += alpha_prob * beta_prob * gamma_dist.pdf(gamma_val, a=alpha, loc=0, scale=1/beta)
            #PostPred_dist['mixture'] += alpha_prob * beta_prob * gamma_dist.pdf(post_pred_val_range, a=alpha, loc=0, scale=1/beta)  # vectorised
                #PostPred_dist.loc[i,'mixture'] += np.log(alpha_prob) + beta_prob + gamma_dist.logpdf(gamma_val, a=alpha, loc=0, scale=1/beta)
#    PostPred_dist['mixture'] = PostPred_dist['mixture'].apply(lambda x: np.exp(x))
    PostPred_dist['mixture'] /= PostPred_dist['mixture'].sum()
    PostPred_dist['single'] = [gamma_dist.pdf(x, a=alpha0, loc=0, scale=1/beta0) for x in post_pred_val_range]
    PostPred_dist['single'] /= PostPred_dist['single'].sum()
    print('Loops took {} sec.'.format(np.round(time.time() - t0, 2)))
    
    plt.figure(figsize=(8,6))
    #PostPred_dist.plot(style=['s-', 'o--'], alpha=.7, figsize=(9,6), fontsize=12)
    plt.plot(post_pred_val_range, PostPred_dist.mixture.values, 'c-', alpha=.7, label='mixture')
    plt.plot(post_pred_val_range, PostPred_dist.single.values, 'k--', alpha=.6, label='single')
    plt.plot(post_pred_val_range, PostPredictive, 'm:', lw=5, alpha=.6, label='mix2')
    plt.legend(fontsize=14)
    plt.xlabel('gamma', fontsize=14)
    plt.title('Gamma posterior predictive', fontsize=16)
    
    """ CAVEATS:
        (1) ranges must be well-picked!
        (2) if vectorisation/broadcasting is not an option, we'd still need to manually construct loops
    """
    