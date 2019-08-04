# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:35:02 2019

@author: Kris
"""

"""
    Following Chapter 15 of CASI (Efron, Hastie) on Large-Scale Hypothesis testing.
    Start with frequentist test, then proceed to apply Variational Bayes.
    
    Dataset of prostate cancer gene expression levels is from textbook's website.
    
    Example Python scritps CASI:
        https://github.com/jrfiedler/CASI_Python
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm
import os


def Bonferroni_bound(N, alpha=.05):
    # N is number of tests; alpha is significance level
    return -norm.ppf(alpha / N)

def FWER(N, alpha=.05):
    pass

def Holms_procedure(p_values, N, alpha=.05):
    p_sorted = np.sort(p_values)
    # to do
    
Pr = pd.read_csv(os.path.join(r'..\..\..\data', 'prostmat.csv')) # loading 'Prostate' dataset
Pr.columns = [c.replace('.','') for c in Pr.columns]

dof = Pr.shape[1] - 2
Pr_z = Pr.apply(lambda t_value: norm.ppf(t.cdf(t_value, dof)))
Pr_z_all = Pr_z.values.reshape(Pr.shape[0]*Pr.shape[1],)
plt.figure(figsize=(7,6))
plt.hist(Pr_z_all, bins=25, density=True)
