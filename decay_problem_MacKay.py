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
    cm. N decays are observed at locations {1.5, 2, 3, 4, 5, 12}
    cm. What is [the distribution of] lambda?
"""

import sys
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import gamma
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='dark')

# Routine for constructing posterior predictive as a mixture of pdfs:
from posterior_predictive_pmf import functional_posterior_predictive_pmf

# FUNCTIONS FOR DECAY PROBLEM
def expon(X, lambd):
    """
        Note: could equivalently use scipy.stats.expon.
        X and/or lambd may be vectors.
    """
    return (1 / lambd) * np.exp(-X / lambd)

def gamma_dist(x, alpha, beta):
    return (beta**alpha) / gamma(alpha) * x**(alpha - 1) * np.exp(-beta * x)

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

# FUNCTIONS FOR CONJ.PRIOR BAYESIAN
def Bayesian_conjugate_inference(obs, alpha0, beta0):
    """
        Using Gamma conjugate prior for Exponential likelihood, see https://en.wikipedia.org/wiki/Conjugate_prior
    """
    #shape = 1
    #alpha0 = shape
    #beta0 = shape * 1  #shape * int(np.round(1. * np.mean(obs), 0))  # prior; can thus choose theta with good start estimate
    alpha = float(alpha0 + len(obs))
    beta = float(beta0 + np.sum(obs))
    return alpha, beta

def get_percentile(X, Y, perc=.05):
    # Assumes X is ordered
    ind = np.nonzero(Y.cumsum()>=perc)[0][0]
    return X[ind]

def compute_likel(lambda_):
    #observation_pmf = dict(zip(x_range, list(naked_likelihood_x['observation prob']) ))
    likelihood_of_x = {}
    for x in x_range:
        likelihood_of_x[x] = expon(x, lambda_) #* observation_pmf[x]
    likelihood_of_x_sum = sum(list(likelihood_of_x.values()))
    likelihood_of_x_norm = {k:v/likelihood_of_x_sum for k,v in likelihood_of_x.items()}
    return likelihood_of_x_norm

def MakeMixturePMF(param_posterior_pmf, likel_pmf_func):
    """ Make a mixture distribution for Bayesian posterior presdictive dist. in 1D.
        Arguments are dict of posterior pmf of parameter 
                      & likelihood function that takes parameter as in input.
        Returns a dict of the normalised mixture pmf.
        Inspired/based on Think Bayes Chapter 5.
    """
    mix = defaultdict(float)
    for param_value, p1 in param_posterior_pmf.items():
        likelihood_x = likel_pmf_func(param_value)
        for x, p2 in likelihood_x.items():
            mix[x] += p1 * p2
    mix_prob_sum = sum(list(mix.values()))
    mix_norm = {k:v/mix_prob_sum for k,v in mix.items()}
    return mix_norm


if __name__=='__main__':
    # =============================================================================
    # M A I N
    # =============================================================================
    x_observations = [1.5, 2, 3, 4, 5, 12]  # ORIGINAL VALUES
    observed_window = (1, 20)

    #x_observations = [2.5, 2.8, 3, 3.5, 4, 5.2, 7, 8, 10.5]  # (indeed only in observation window)
    #true_lambda = 5
    #n_samples = 20
    #all_observations = np.random.exponential(scale=true_lambda, size=n_samples)
    #observed_window = (2, 12)
    #x_observations = [q for q in all_observations if q>=observed_window[0] and q<=observed_window[1]]

    x_range = np.linspace(0, 50, 50*10+1)
    lambda_range = np.linspace(.005, 100, 1000)  # need fine grid to get smooth curves on low end of log scale
    #lambda_prior_extremes = (.1, 15)  # for uniform prior
    lambda_prior_normal = norm(np.mean(x_observations),3)
    x_observation_window = observation_prob(x_range, Xmin=observed_window[0], Xmax=observed_window[1])
    
    # =============================================================================
    # P(x|lambda) as a function of x  (for hypothetical lambdas)
    # =============================================================================
    lambda_examples = [2, 5, 10] #[.1, .2, .3]
    naked_likelihood_x = pd.DataFrame(data=x_range, columns=['x'])
    naked_likelihood_x['observation prob'] = observation_prob(x_range)
    naked_likelihood_x['observation prob'] /= naked_likelihood_x['observation prob'].sum()
    
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1, 1, 1)
    #naked_likelihood_x[['x','observation prob']].plot(x='x', y='observation prob', figsize=(7,5), grid=True, lw=3)
    ax.fill_between(list(observed_window), [0,0], [.025,.025], color='lightblue', alpha=.5)
    ax.vlines(x_observations, 0, .015, 'k', linestyles='solid', label='observations')
    ax.set_title('Observation window', fontsize=15)
    ax.set_xlabel('x', fontsize=15)
    ax.tick_params(axis='y',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False) # labels along the bottom edge are off
    #plt.ylabel('P(x is observed)')
    ax.set_xlim([0,35])
    ax.set_ylim([0,.02])
    ax.legend(fontsize=15)
    ax.tick_params(axis='x', which='major', labelsize=13)
    ax.get_yaxis().set_ticks([])
    
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1, 1, 1)
    #plt.plot(x_range, naked_likelihood_x['observation prob'], 'k--', label='observation prob.')
    for i,lam in enumerate(lambda_examples):
        likel_of_x = expon(x_range, lam) * naked_likelihood_x['observation prob']
        naked_likelihood_x['lambda_'+str(i)] = likel_of_x / np.sum(likel_of_x)
        ax.plot(x_range, naked_likelihood_x['lambda_'+str(i)], label='$\lambda$='+str(lam))
    ax.legend(fontsize=13)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('P(x | $\lambda$)', fontsize=15)
    ax.set_title('Likelihood of x (example values)', fontsize=16)
    ax.set_xlim([0,35])
    ax.tick_params(axis='x', which='major', labelsize=13)
    
    # =============================================================================
    # P(x|lambda) as a function of lambda (for hypothetical x values)
    # =============================================================================
    naked_likelihood_lam = pd.DataFrame(data=lambda_range, columns=['lambd'])
    x_examples = [3, 5, 12, 27]  # have to lie in observation window in order to be non-trivial
    fig = plt.figure(figsize=(8,4))
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
    ax.set_xlim(.25, 100)
    ax.legend(fontsize=13)
    ax.set_xlabel('$\lambda$', fontsize=15)
    ax.set_ylabel('P(x | $\lambda$)', fontsize=15)
    ax.set_title('Likelihood of $\lambda$ (example values)', fontsize=16)
    ax.tick_params(axis='x', which='major', labelsize=13)
    
    # =============================================================================
    # Compute likelihood of lambda P(x|lambda) for given x observations and obs. window
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
# =============================================================================
# =============================================================================
# #     TO DO: CHECK NORMALISING CONSTANTS IN ^^^^
# =============================================================================
# =============================================================================

    # plotting this
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(lambda_range, final_likelihood_lam['all x'], 'k-', lw=2.5, label='all x')
    ax.plot(lambda_range, np.zeros_like(lambda_range), 'k--', alpha=.5)
    ax.set_xlim(0,50)
    ax.legend(fontsize=15)
    ax.set_xlabel('$\lambda$', fontsize=15)
    ax.set_ylabel('P({x} | $\lambda$)', fontsize=15)
    ax.set_title('Likelihood of $\lambda$ for given dataset', fontsize=16)
    ax.tick_params(axis='x', which='major', labelsize=13)
    
    # Idem, on a log scale (as in McKay book)
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(lambda_range, final_likelihood_lam['all x'], 'k-', lw=2.5, label='all x')
    ax.plot([.001, 100], [0, 0], 'k--', alpha=.5)
    ax.set_xscale('log')
    ax.set_xlim(0.25,100)
    ax.legend(fontsize=15)
    ax.set_xlabel('$\lambda$', fontsize=15)
    ax.set_ylabel(r'P({x} | $\lambda$)', fontsize=15)
    ax.set_title(r'Likelihood of $\lambda$ for given dataset', fontsize=16)
    ax.tick_params(axis='x', which='major', labelsize=13)

    # =============================================================================
    #    LAMBDA posteriors: with and without* CENSORING APPLIED (* USING CONJ.PRIOR)
    # =============================================================================
    # [0A] Construct flat uniform prior
    #lambda_prior = np.zeros_like(lambda_range)  # Lmin < L < Lmax assumed; cf. first lambda sample plot
    #lambda_prior[(lambda_range>=lambda_prior_extremes[0]) & (lambda_range<=lambda_prior_extremes[1])] = 1
    #lambda_prior[(lambda_range>=lambda_prior_extremes[0]) & (lambda_range<=lambda_prior_extremes[1])] /= np.sum(lambda_prior)
    # [0B]
    lambda_prior = lambda_prior_normal.pdf(lambda_range)
    lambda_prior = gamma_dist(lambda_range, alpha=2, beta=1/2)
    lambda_prior /= lambda_prior.sum()

    # [1] Censored case:
    lambda_posterior_censored = lambda_prior * final_likelihood_lam['all x'].values
    lambda_posterior_censored /= np.sum(lambda_posterior_censored)
    lambda_posteriors = {}
    lambda_posteriors['censored'] = (get_percentile(lambda_range, lambda_posterior_censored, .2), \
                                     np.dot(lambda_range, lambda_posterior_censored), \
                                     get_percentile(lambda_range, lambda_posterior_censored, .8))
    
    # [2] Uncensored: Compute lambda posterior via exponential likelihood
    lambda_posterior_uncensored = lambda_prior.copy()
    for x in x_observations:
        lambda_posterior_uncensored /= np.sum(lambda_posterior_uncensored)
        for i,lambd in enumerate(lambda_range):
            lambda_posterior_uncensored[i] *= expon(x, lambd)
    lambda_posterior_uncensored /= np.sum(lambda_posterior_uncensored)
    lambda_posteriors['uncensored'] = (get_percentile(lambda_range, lambda_posterior_uncensored, .2), \
                                       np.dot(lambda_range, lambda_posterior_uncensored), \
                                       get_percentile(lambda_range, lambda_posterior_uncensored, .8)) 

    # Plot posterior results
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(lambda_range, lambda_prior, 'k--', lw=2, alpha=.2, label='prior')
    ax.plot(lambda_range, lambda_posterior_censored, 'm-', label='with censoring')
    ax.plot(lambda_range, lambda_posterior_uncensored, 'c-', label='without')
    #ax.plot([lambda_posteriors['censored'][1]]*2, [0,.0150], 'm--')
    #ax.plot([lambda_posteriors['uncensored'][1]]*2, [0,.0150], 'c--')
    #ax.set_xscale('log')
    ax.set_xlim(0,20)
    ax.legend(fontsize=15)
    ax.set_xlabel('$\lambda$', fontsize=15)
    ax.set_ylabel('P($\lambda$ | {x})', fontsize=15)
    ax.set_title('$\lambda$ posterior', fontsize=16)
    ax.tick_params(axis='x', which='major', labelsize=13) 
    
    # =============================================================================
    # Posterior predictives (1) POINT ESTIMATES FROM POSTERIOR
    # =============================================================================
    Censored_post_pred = {'low': expon(x_range, lambda_posteriors['censored'][0]), \
                          'mean': expon(x_range, lambda_posteriors['censored'][1]), \
                          'high': expon(x_range, lambda_posteriors['censored'][2])}

    Uncensored_post_pred = {'low': expon(x_range, lambda_posteriors['uncensored'][0]), \
                            'mean': expon(x_range, lambda_posteriors['uncensored'][1]), \
                            'high': expon(x_range, lambda_posteriors['uncensored'][2])}
    
#    # MEANS in original x space --> TO DO: REPLACE MEANS BY 
#    fig = plt.figure(figsize=(8,4))
#    ax = fig.add_subplot(1, 1, 1)
#    ax.fill_between(list(observed_window), [0,0], [.12,.12], color='lightblue', alpha=.5)
#    ax.plot(x_range, Censored_post_pred['mean'], 'b-', label='censored')
#    ax.plot(x_range, Censored_post_pred['low'], 'b--', label='censored')
#    ax.plot(x_range, Uncensored_post_pred['mean'], 'r-', label='uncensored')
#    ax.legend(fontsize=13)
#    ax.set_xlabel('x', fontsize=15)
#    ax.set_ylabel("P(x | x')", fontsize=15)
#    ax.set_title('Posterior predictives', fontsize=16)
#    ax.set_xlim([0,35])
#    ax.set_ylim([0,.12])
#    ax.tick_params(axis='x', which='major', labelsize=13)
#    
#    # CDFs plot
#    fig = plt.figure(figsize=(8,4))
#    ax = fig.add_subplot(1, 1, 1)    
#    ax.fill_between(x_range, Censored_post_pred['low'].cumsum()/Censored_post_pred['low'].sum(), \
#                    Censored_post_pred['high'].cumsum()/Censored_post_pred['high'].sum(), color='b', alpha=.15)
#    ax.plot(x_range, Censored_post_pred['mean'].cumsum()/Censored_post_pred['mean'].sum(), 'b-', label='censored')
#
#    ax.fill_between(x_range, Uncensored_post_pred['low'].cumsum()/Uncensored_post_pred['low'].sum(), \
#                    Uncensored_post_pred['high'].cumsum()/Uncensored_post_pred['high'].sum(), color='r', alpha=.15)
#    ax.plot(x_range, Uncensored_post_pred['mean'].cumsum()/Uncensored_post_pred['mean'].sum(), 'r-', label='uncensored')
#    ax.set_xlim([0,20])
#    ax.set_ylim([0,1])    

    # =============================================================================
    # Posterior predictives (2) FULL MIXTURES (Think Bayes' PMF loop method)
    # =============================================================================
    # Censored pred.post.
    posterior_pmf_censored = dict(zip(list(lambda_range), list(lambda_posterior_censored)))
    posterior_predictive_censored = MakeMixturePMF(posterior_pmf_censored, compute_likel)
    # Uncensored pred.post.
    posterior_pmf_uncensored = dict(zip(list(lambda_range), list(lambda_posterior_uncensored)))
    posterior_predictive_uncensored = MakeMixturePMF(posterior_pmf_uncensored, compute_likel)
    # Plot PDFs
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1, 1, 1)
    #ax.fill_between(list(observed_window), [0,0], [.03,.03], color='lightblue', alpha=.3)
    ax.plot(list(posterior_predictive_censored.keys()), np.array(list(posterior_predictive_censored.values()), dtype=float).cumsum(), 'm-', label='censored')
    ax.plot(list(posterior_predictive_uncensored.keys()), np.array(list(posterior_predictive_uncensored.values()), dtype=float).cumsum(), 'c-', label='uncensored')
#    ax.plot(x_range, Censored_post_pred['mean']/Censored_post_pred['mean'].sum(), 'm--', alpha=.4, label='censored mean')
#    ax.plot(x_range, Uncensored_post_pred['mean']/Uncensored_post_pred['mean'].sum(), 'c--', alpha=.4, label='uncensored mean')
    ax.legend(fontsize=13)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel("P(x | x')", fontsize=15)
    ax.set_title('Posterior predictives PDF', fontsize=16)
    ax.set_xlim([0,20])
    ax.set_ylim([0,1])
    ax.tick_params(axis='x', which='major', labelsize=13)  
    
# =============================================================================
#     # TO DO COMPARE ^^ WITH POST.PRED. BASED ON MEAN LAMBDAs
# =============================================================================
