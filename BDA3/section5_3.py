# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 16:13:38 2019

@author: Kris
"""

""" Recreating the example in section 3.7 of <Bayesian Data Analysis> by Gelman et al. (BDA3)
    A rather simple case of 2-parameter multivariate hierarchical inference for a regression model based on 
      non-conjugate analysis, solved by grid simulations (without using MCMC).
"""

import numpy as np
import pandas as pd
#from scipy.stats import gamma, binom
from scipy.special import logit, expit  # logistic transformation and its inverse, resp.
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# =============================================================================
# =============================================================================
# =============================================================================
# # # TO DO!!!!!
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # CODE BELOW IS SECTION 3.7
# =============================================================================
# =============================================================================


def get_data():
    """ See table 3.1 on page 74
        index is group number.    
        columns represent resp. dose, number of animals and number of deaths.
    """
    data  = pd.DataFrame(columns=list('xny'), \
                         data=np.array([[-.86, 5, 0], [-.3, 5, 1], [-.05, 5, 3], [.76, 5, 5]]), \
                         index=range(1,5))
    data.index.name = 'i'
    return data

def likel(alpha, beta):
    """ alpha, beta are (sampled or fixed) scalars; uses data vectors of same size (N,1) from global dataframe df """
    L = 1
    for _,row in df.iterrows():
        L *= ((expit(alpha + beta*row.x))**row.y) * (1 - expit(alpha + beta*row.x))**(row.n - row.y)
    return L

def log_likel(alpha, beta):
    """ alpha, beta are scalars or vectors.
        data is contained in global variable dataframe df.
        to do: fully vectorised/broadcasted.
    """
    LL = 0
    for _,row in df.iterrows():
        A = alpha + beta*row.x
        LL += row.y * (A - np.log(1 + np.exp(A))) \
              - (row.n - row.y) * np.log(1 + np.exp(A))
    return LL

def likel_for_optimisation(V):
    """ for finding MLE used as initial values of sampling.
        V[0] is alpha, V[1] is beta
    """
    #LL = 0
    #for _,row in df.iterrows():
    #    LL += row.y*np.log(expit(V[0] + V[1]*row.x)) + (row.n - row.y)*np.log(1 - expit(V[0] + V[1]*row.x))
    L = 1
    for _,row in df.iterrows():
        L *= ((expit(V[0] + V[1]*row.x))**row.y) * (1 - expit(V[0] + V[1]*row.x))**(row.n - row.y)
    return -L


if __name__=='__main__':
    
    # prepare
    n_samples = 1000
    df = get_data()
    df['theta'] = df['y'] / df['n']
    df['theta_logit'] = logit(df['theta'])

    # MLE estimate of hyperparameters (to do: why cannot use log likelihood here?)
    res = minimize(likel_for_optimisation, (1, 1), options={'maxiter':100})
    print('\nMLE estimate (alpha,beta):', res.x)

    # Find contours of (unnormalized) joint posterior density of hyperparams (alphas, beta)
    # Note that this the resulting points are not suitable as posterior samples!
    alpha0 = np.linspace(-5, 10, 801)
    beta0 = np.linspace(-10, 40, 1001)
    subgrid_stepsize = {'alpha':np.diff(alpha0)[0], 'beta':np.diff(beta0)[0]}
    av, bv = np.meshgrid(alpha0, beta0)
    #L_vals = np.empty_like(av)
    Log_LH_vals = np.empty_like(av)
    for i,zz in enumerate(zip(av, bv)):
        #L_vals[i,:] = likel(zz[0], zz[1])  # to do: replace by np.prod()
        Log_LH_vals[i,:] = log_likel(zz[0], zz[1])  # log of likelihood (=posterior here)
    Log_LH_vals -= Log_LH_vals.max().max()
    L_vals = np.exp(Log_LH_vals)
    plt.figure(figsize=(6,5))
    h = plt.contourf(av, bv, L_vals)
    plt.xlabel('alpha', fontsize=15)
    plt.ylabel('beta', fontsize=15)
    plt.title('Joint posterior of hyperparameters', fontsize=16)
    
    # Compute marginal posterior of alpha ('Step 1')
    joint_pdf = L_vals / L_vals.sum()  # approximated log normalised posterior (log to )
    marginal_posterior_alpha = joint_pdf.sum(axis=0)
    marginal_posterior_alpha /= marginal_posterior_alpha.sum()
    plt.figure(figsize=(6,4))
    plt.plot(alpha0, marginal_posterior_alpha,'g-')
    plt.xlabel('alpha', fontsize=15)
    plt.ylabel('p(alpha | y)', fontsize=15)
    plt.title('Marginal posterior of alpha', fontsize=16)

    # 'Step 2a': draw from p(alpha|y) by using inverse discrete CDF
    alpha_cdf = marginal_posterior_alpha.cumsum()
    alpha_cdf /= alpha_cdf.max()  # to fix potential rounding off errors
    plt.figure(); plt.plot(alpha0, alpha_cdf, 'k-'); plt.title('CDF of p(alpha|y)', fontsize=16); plt.xlabel('alpha',fontsize=15)
    #us = np.random.choice(marginal_posterior_alpha, size=1000)
    #uniform_samples = np.random.uniform(0, 1, size=1000)
    #alpha_samples = np.quantile(alpha0, us, interpolation='linear') # FAIL: should be able to use this function for Inverse cdf sampling from arbitrary discrete distribution! to do: do all this in pandas df    
    alpha_samples = np.random.choice(alpha0, size=n_samples, replace=True, p=marginal_posterior_alpha)
    plt.figure(figsize=(6,4))
    plt.hist(alpha_samples, density=True, bins=35)
    plt.title('alpha samples', fontsize=15)
    
    # 'Step 2b': draw beta samples from the conditional posterior p(beta|alpha, y), given alpha samples
    beta_samples = np.empty_like(alpha_samples)
    for i,a in enumerate(alpha_samples):
        alpha_index = np.nonzero(alpha0==a)[0][0]
        beta_conditional_post = joint_pdf[:,alpha_index] / joint_pdf[:,alpha_index].sum()  # p(beta | alpha, y), i.e. gives beta post.pdf as a slice of joint pdf.
        beta_samples[i] = np.random.choice(beta0, size=1, p=beta_conditional_post)
    plt.figure(figsize=(6,4))
    plt.hist(beta_samples, density=True, color='m', bins=35)
    plt.title('beta samples', fontsize=15)

    # 'Step 2c': Add uniformly random subgrid jitter to smooth out discretely sampled values
    alpha_jitter = np.random.uniform(low=-subgrid_stepsize['alpha']/2, \
                                     high=subgrid_stepsize['alpha']/2, \
                                     size=n_samples)
    alpha_samples += alpha_jitter
    beta_jitter = np.random.uniform(low=-subgrid_stepsize['beta']/2, \
                                    high=subgrid_stepsize['beta']/2, \
                                    size=n_samples)
    beta_samples += beta_jitter

    # Plot resulting posterior samples, on top of joint pdf
    plt.figure(figsize=(8,7))
    h = plt.contourf(av, bv, L_vals, alpha=.7)
    plt.plot(alpha_samples, beta_samples, 'k.', markersize=4, alpha=.5)
    plt.title('Samples from posterior PDF of hyperparameters', fontsize=16)
    plt.xlabel('alpha', fontsize=15)
    plt.ylabel('beta', fontsize=15)
    plt.xlim([-3, 7])
    plt.ylim([-3, 35])
        