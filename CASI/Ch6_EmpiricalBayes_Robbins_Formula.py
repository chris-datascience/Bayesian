# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:28:58 2019

@author: Kris
"""


"""
    Following Chapter 6 of CASI (Efron, Hastie) on Emperical Bayes.
    Showing Robbins' Formula (Section 6.1).
    Derivations are described in section 6.1.
    Alternative code:  https://github.com/jrfiedler/CASI_Python/blob/master/chapter06/ch06s01.ipynb
"""

import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt

def Poisson(theta, x):
    rv = poisson(theta)    
    return rv.pmf(x)

def Robbins(x, func):
    # Equation (6.5); func is a function f (that supports vectorisation)
    return (x + 1) * func(x + 1) / func(x)

def get_proportion(n):
    # input is index = claim category
    try:
        return Claims.loc[n,'counts_year0'] / Claims['counts_year0'].sum()
    except:
        return 0
    
    
if __name__=='__main__':
    #    Suppose number of claims per account holder follows Poisson dist.
    #    Example:
    x = range(9)
    P = Poisson(2.3, x)
    plt.figure(figsize=(7,4))
    plt.vlines(x, 0, P, colors='b', linestyles='-', lw=3, label='frozen pmf')
    plt.xlabel('claims', fontsize=14)
    plt.ylabel('Expected claims = parameter theta = 2.3', fontsize=15)

    # Data
    Claims = pd.DataFrame(columns=['counts_year0'], index=range(8), data=[7840, 1317, 239, 42, 14, 4, 4, 1])
    Claims.index.name = 'claims' 

    # Estimate marginal density of x (number of claims) by proportion of total counts in category x in year0:
    for claims, row in Claims.iterrows():
        Claims.loc[claims,'Robbins_year1'] = Robbins(claims, get_proportion).round(3)
    
    print(Claims.T)

    # Apply Gamma regression to smooth out priors:
    # --TO DO--
    #Claims['Robbins_Gamma_year1'] = 