# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 21:31:15 2019

@author: erdbrca
"""

"""
    Numpy / Pandas version of Decay Problem appearing in
    Section 3.3 in MacKay's <Information Theory..> textbook
    
    adapted from code in Think Bayes
    Copyright 2010 Allen B. Downey
    License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html

    This file contains a partial solution to a problem from
    MacKay, "Information Theory, Inference, and Learning Algorithms."

    Problem formulation
    <<< Unstable particles are emitted from a source and decay at a
    distance x, a real number that has an exponential probability
    distribution with [parameter] lambda.  Decay events can only be
    observed if they occur in a window extending from x=1 cm to x=20
    cm. N decays are observed at locations { 1.5, 2, 3, 4, 5, 12}
    cm. What is [the distribution of] lambda?
"""

import sys
import numpy as np
import pandas as pd
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='dark')

# Routine for constructing posterior predictive as a mixture of pdfs:
from posterior_predictive_pmf import functional_posterior_predictive_pmf


def expon(X, lambd):
    """
        Note: could equivalently use scipy.stats.expon.
        X and/or lambd may be vectors.
    """
    return (1 / lambd) * np.exp(-X / lambd)

def expon_integral(lambd, xmin, xmax):
    """
        Serves as normalisation in likelihoods.
    """
    return np.exp(-xmin / lambd) - np.exp(-xmax / lambd)

def lomax(X, alpha, beta):
    """
        Note: could equivalently use scipy.stats.lomax.
    """
    return (alpha / beta) * ((1 + X / beta)**(-(alpha + 1)))

def observation_prob(X, Xmin=1, Xmax=20):
    """
        Prob.Dens.Function of Observation window; to take into account observational bias.
        Is used to compute likelihood given the data.
        Make sure this is vectorised, i.e. input should be np.array
    """
    Y = np.zeros_like(X)
    Y[(X>=Xmin) & (X<=Xmax)] = 1
    return Y


if __name__=='__main__':
    # =============================================================================
    # M A I N
    # =============================================================================
    x_observations = [1.5, 2, 3, 4, 5, 12]  # indeed only in observation window
    observed_window = (1, 20)
    x_range = np.linspace(0, 50, 50*5+1)
    lambda_range = np.linspace(.002, 150, 3001)  # need fine grid to get smooth curves on low end of log scale
    x_observation_window = observation_prob(x_range, Xmin=observed_window[0], Xmax=observed_window[1])
    
    # =============================================================================
    # P(x|lambda) as a function of x  (for hypothetical lambdas)
    # =============================================================================
    lambda_examples = [2, 5, 10] #[.1, .2, .3]
    naked_likelihood_x = pd.DataFrame(data=x_range, columns=['x'])
    naked_likelihood_x['observation prob'] = observation_prob(x_range)
    naked_likelihood_x['observation prob'] /= naked_likelihood_x['observation prob'].sum()
    
    plt.figure()
    naked_likelihood_x[['x','observation prob']].plot(x='x', y='observation prob', figsize=(7,5), lw=3)
    plt.title('Observation window')
    plt.xlabel('x')
    plt.ylabel('P(x is observed)')
    
    plt.figure(figsize=(9,5))
    #plt.plot(x_range, naked_likelihood_x['observation prob'], 'k--', label='observation prob.')
    for i,lam in enumerate(lambda_examples):
        likel_of_x = expon(x_range, lam) * naked_likelihood_x['observation prob']
        naked_likelihood_x['lambda_'+str(i)] = likel_of_x / np.sum(likel_of_x)
        plt.plot(x_range, naked_likelihood_x['lambda_'+str(i)], label='lambda='+str(lam))
    plt.legend(fontsize=13)
    plt.xlabel('x') 
    plt.ylabel('P(x|lambda)')
    plt.title('Likelihood of x')
    
    # =============================================================================
    # P(x|lambda) as a function of lambda (for hypothetical x values)
    # =============================================================================
    naked_likelihood_lam = pd.DataFrame(data=lambda_range, columns=['lambd'])
    x_examples = [3, 5, 12, 27]  # have to lie in observation window in order to be non-trivial
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(1, 1, 1)
    for i,x in enumerate(x_examples):
        # [option 1] Using prefab analytical integration constant:
        #Z = expon_integral(lambda_range, observed_window[0], observed_window[1])
        
        # [option 2] Using on-the-spot integration (which will hopefully work for more complex functions as well):
        Z = np.array([quad(expon, observed_window[0], observed_window[1], args=(lambd))[0] for lambd in lambda_range])

        likel_of_lambda = (expon(x, lambda_range) / Z ) * observation_prob(x)  # filter by observation window
        naked_likelihood_lam['x_'+str(i)] = likel_of_lambda
        ax.plot(lambda_range, naked_likelihood_lam['x_'+str(i)], label='x='+str(x))
    ax.set_xscale('log')
    ax.set_xlim(.25, 125)
    ax.legend(fontsize=13)
    ax.set_xlabel('lambda')
    ax.set_ylabel('P(x|lambda)')
    ax.set_title('Likelihood of lambda')
    
    # =============================================================================
    # Compute likelihood of lambda P(x|lambda) for actual observations of x
    # =============================================================================
    final_likelihood_lam = pd.DataFrame(data=lambda_range, columns=['lambd'])
    final_likelihood_lam['all x'] = 1
    for i,x in enumerate(x_observations):
        Z = expon_integral(lambda_range, observed_window[0], observed_window[1])
        final_likelihood = (expon(x, lambda_range) / Z) * observation_prob(x) # In present binary observation case, this is trivial
        final_likelihood_lam['x_'+str(i)] = final_likelihood
        #plt.plot(lambda_range, final_likelihood_lam['x_'+str(i)], lw=1, alpha=.7, label='x='+str(x))
        final_likelihood_lam['all x'] *= final_likelihood_lam['x_'+str(i)]
    final_likelihood_lam['all x'] /= np.sum(final_likelihood_lam['all x'])
    plt.figure(figsize=(9,5))
    plt.plot(lambda_range, final_likelihood_lam['all x'], 'k-', lw=2.5, label='all x')
    plt.plot(lambda_range, np.zeros_like(lambda_range), 'k--', alpha=.5)
    plt.xlim(0,50)
    plt.legend(fontsize=15)
    plt.xlabel('lambda')
    plt.ylabel('P(x|lambda)')
    plt.title('Likelihood of lambda')
    # Idem, on a log scale (as in McKay book)
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(lambda_range, final_likelihood_lam['all x'], 'k-', lw=2.5, label='all x')
    ax.plot(lambda_range, np.zeros_like(lambda_range), 'k--', alpha=.5)
    ax.set_xscale('log')
    plt.xlim(0.25,100)
    plt.legend(fontsize=15)
    plt.xlabel('lambda')
    plt.ylabel('P(x|lambda)')
    plt.title('Likelihood of lambda')
    
    
    # =============================================================================
    # Compute posterior predictive distribution of corrected (estimated) decay
    # =============================================================================
    #   eeehhhh. we assume a flat prior, in fact we're ignoring Bayes rule (simply taking the corrected likelihood)
    assumed_true_lamba = .25
    #psoterior_predictive_mix_pmf = compute_posterior_predictive_pmf
    
    #naked_likelihood_x = pd.DataFrame(data=x_range, columns=['x'])
    #naked_likelihood_x['observation prob'] = observation_prob(x_range)
    #naked_likelihood_x['observation prob'] /= naked_likelihood_x['observation prob'].sum()
    #plt.figure(figsize=(9,5))
    #plt.plot(x_range, naked_likelihood_x['observation prob'], 'k--', label='observation prob.')
    #for i,lam in enumerate(lambda_examples):
    #    naked_likelihood_x['lambda_'+str(i)] = expon(x_range, lam) * naked_likelihood_x['observation prob']
    #    naked_likelihood_x['lambda_'+str(i)] /= np.sum(naked_likelihood_x['lambda_'+str(i)])
    #    plt.plot(x_range, naked_likelihood_x['lambda_'+str(i)], label='lambda='+str(lam))
    #plt.legend()
    #plt.xlabel('x') 
    #plt.ylabel('P(x|lambda)')
    #plt.title('Likelihood of x')