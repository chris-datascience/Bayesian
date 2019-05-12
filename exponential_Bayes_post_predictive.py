# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 21:05:51 2019

@author: erdbrca
"""
from scipy.stats import gamma, lomax
import numpy as np
import matplotlib.pyplot as plt


def CI_theta(obs):
    mu = np.mean(obs)
    stdev = np.sqrt((mu**2) / float(len(obs)))
    return mu, stdev

def Bayesian_conjugate_inference(obs):
    """
        Using Gamma conjugate prior for Exponential likelihood, see https://en.wikipedia.org/wiki/Conjugate_prior
    """
    shape = 3
    alpha0 = shape
    beta0 = shape * 5  #shape * int(np.round(1. * np.mean(obs), 0))  # prior; can thus choose theta with good start estimate
    alpha = float(alpha0 + len(obs))
    beta = float(beta0 + np.sum(obs))
    return alpha, beta

#    return np.mean(data), np.percentile(data, 2.5), np.percentile(data, 97.5), alpha, beta    

def lomax_manual(x, alpha, lambd):
    return (alpha/lambd) * ((1 + x / lambd)**(-(alpha + 1)))
    
def MakeMixture(metapmf, label='mix'):
    """Make a mixture distribution.
    Args:
      metapmf: Pmf that maps from Pmfs to probs.
      label: string label for the new Pmf.
    Returns: Pmf object.
    """
    mix = {} #Pmf(label=label)
    for pmf, p1 in metapmf.Items():
        for x, p2 in pmf.Items():
            mix[x] += p1 * p2
    return mix


if __name__=='__main__':
    true_theta = .25  # N.B. corresponds to expected value of 1/true_theta
    N_samples = 1000
    Z_ = 1.96  # 95% confidence
    
    # Draw from an exponential dist. and try to recover main parameter
    Observations = np.random.exponential(scale=1/true_theta, size=(N_samples,))
    
    # Compute Confindence interval of theta from observations:
    mu, sigma = CI_theta(Observations)
    freq_mean = 1. / mu
    CI_high = 1 / (mu + Z_ * sigma)
    CI_low = 1 / (mu - Z_ * sigma)
    
    # Compute theta posterior, expressed by Gamma parameters alpha and beta
    alpha_, beta_ = Bayesian_conjugate_inference(Observations)
    
    theta_samples = gamma.rvs(alpha_, loc=0, scale=1./float(beta_), size=10000)
    theta_range = np.linspace(0,1,500)
    g1 = gamma(alpha_, loc=0, scale=1./float(beta_))
    plt.figure(figsize=(8,5))
    plt.hist(theta_samples, bins=55, density=True, alpha=.5)
    plt.plot(theta_range, g1.pdf(theta_range), 'b-')
    plt.xlim(np.min(theta_samples), np.max(theta_samples))
    plt.title('Theta posterior: samples versus pdf', fontsize=15)
    plt.xlabel('theta (-)', fontsize=15)

    # Compute predictive posterior directly from analytical formula: the Lomax distribution
    # See https://en.wikipedia.org/wiki/Lomax_distribution
    # and https://en.wikipedia.org/wiki/Conjugate_prior
    X = np.linspace(0, 100, 10000) # fixed sample locations
    PredPost = lomax_manual(X, alpha_, beta_)
    PredPost = PredPost / np.sum(PredPost)
    
    ll = lomax(c=alpha_, scale=float(beta_))
    PredPost2 = ll.pdf(X)
    PredPost2 /= np.sum(PredPost2)
    print(np.allclose(PredPost2, PredPost))
    
    true_pdf = true_theta*np.exp(-true_theta*X)
    true_pdf /= np.sum(true_pdf)
    
    print('True mean of exponential: %1.3f' % (1. / true_theta))
    print('Estimated mean from pred.post.: %1.3f' % (np.dot(X, PredPost / np.sum(PredPost))))
    
    # Now let's try to replicate this posterior predictive from the (discrete) prob. mass function of
    # the theta parameter, namely by constructing a mixture of exponentials, not using
    # a 'meta-pdf' structure as outlined in <<Think Bayes>> (essentially based on the essential construct
    # of a nested dictionary and bypassing all classes etc., see MakeMixture() function)), but instead with a nested for-loop.
    g = gamma(alpha_, loc=0, scale=1./float(beta_))
    theta_probs = g.pdf(theta_range)  # vectorised evaluation
    Mix_pmf = np.zeros((len(X), ), dtype=float)
    for i,x in enumerate(X):
        for theta, P in zip(theta_range, theta_probs):
            exponential = theta * np.exp(-theta * x)
            Mix_pmf[i] += exponential * P
    Mix_pmf /= np.sum(Mix_pmf)

    # == PLOTTING it all ==
    plt.figure(figsize=(10,7))
    plt.plot(X, PredPost, 'g-', lw=3, label='lomax posterior predictive')
    plt.plot(X, true_pdf, 'k--', lw=2, alpha=.7, label='true')
    plt.plot(X, Mix_pmf, 'r-', lw=5, alpha=.3, label='mixture pdf')
    plt.xlim(0, 20)
    plt.legend(fontsize=15, loc='best')

# =============================================================================
#     CONCLUSION: LOMAX MATCHES WITH MANUALLY COMPUTED MIXTURE OF EXPONENTIALS (a la "THINK BAYES META_PDF")
# =============================================================================

