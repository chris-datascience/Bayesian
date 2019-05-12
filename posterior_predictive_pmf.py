# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 16:19:29 2019

@author: erdbrca
"""

"""
    1D PostPred PDF approx. by PMF, a la <<Think Bayes>>, see pg. 47, 87
    also see demo in <<exponential_Bayes_post_predictive.py>>
"""

import numpy as np


def functional_posterior_predictive_pmf(param_range, like_range, param_pmf_function, likelihood_pmf_function):
    """
        Posterior Predictive distribution USING FUNCTIONS
        like_range = range in final pred.posterior (in same space as likelihood!)
        param_range = range in posterior/prior parameter space, assuming 1 parameter only
        param_pmf = pmf function of parameter, derived via conjugate prior or otherwise, assumed normalised (but not essential perhaps?!)
        likelihood_pmf = pmf function of likelihood, for given parameter theta
    """
    theta_probs = param_pmf_function(param_range)  # vectorised evaluation
    Mix_pmf = np.zeros((len(like_range), ), dtype=float)
    for i,x in enumerate(like_range):
        for theta, P in zip(param_range, theta_probs):
            likeli = likelihood_pmf_function(theta, x)
            Mix_pmf[i] += likeli * P
    Mix_pmf /= np.sum(Mix_pmf)
    return Mix_pmf

def dict_posterior_predictive_pmf(param_pmf_dict, likelihood_pmf_dict):
    """
        Posterior Predictive distribution USING DICTS
        like_range = range in final pred.posterior (in same space as likelihood!)
        param_range = range in posterior/prior parameter space, assuming 1 parameter only
        param_pmf = pmf function of parameter, derived via conjugate prior or otherwise, assumed normalised (but not essential perhaps?!)
        likelihood_pmf = pmf function of likelihood, for given parameter theta
    """
    # TO DO
#    theta_probs = param_pmf_function(param_range)  # vectorised evaluation
#    Mix_pmf = np.zeros((len(like_range), ), dtype=float)
#    for i,x in enumerate(like_range):
#        for theta, P in zip(param_range, theta_probs):
#            likeli = likelihood_pmf_function(theta, x)
#            Mix_pmf[i] += likeli * P
#    Mix_pmf /= np.sum(Mix_pmf)
#    return Mix_pmf
#    