# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:10:33 2019

@author: erdbrca
"""

import numpy as np
import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt

"""
    Bayes supersedes T-test (BEST) demo from <<BMH>> textbook.
    Source:
        https://docs.pymc.io/notebooks/BEST.html
"""

drug = [101,100,102,104,102,97,105,105,98,101,100,123,105,103,100,95,102,106,
        109,102,82,102,100,102,102,101,102,102,103,103,97,97,103,101,97,104,
        96,103,124,101,101,100,101,101,104,100,101]
placebo = [99,101,100,101,102,100,97,101,104,101,102,102,100,105,88,101,100,
           104,100,100,100,101,102,103,97,101,101,100,101,99,101,100,100,
           101,100,99,101,100,102,99,100,99]
y1 = np.array(drug)
y2 = np.array(placebo)
y = pd.DataFrame(dict(value=np.r_[y1, y2], group=np.r_[['drug']*len(drug), ['placebo']*len(placebo)]))
y.hist('value', by='group');

μ_m = y.value.mean()
μ_s = y.value.std() * 2
σ_low = 1
σ_high = 10

with pm.Model() as model:
    group1_mean = pm.Normal('group1_mean', μ_m, sd=μ_s)
    group2_mean = pm.Normal('group2_mean', μ_m, sd=μ_s)

    group1_std = pm.Uniform('group1_std', lower=σ_low, upper=σ_high)
    group2_std = pm.Uniform('group2_std', lower=σ_low, upper=σ_high)        
    
    ν = pm.Exponential('ν_minus_one', 1/29.) + 1

    λ1 = group1_std**-2
    λ2 = group2_std**-2

    group1 = pm.StudentT('drug', nu=ν, mu=group1_mean, lam=λ1, observed=y1)
    group2 = pm.StudentT('placebo', nu=ν, mu=group2_mean, lam=λ2, observed=y2)

    diff_of_means = pm.Deterministic('difference of means', group1_mean - group2_mean)
    diff_of_stds = pm.Deterministic('difference of stds', group1_std - group2_std)
    effect_size = pm.Deterministic('effect size',  diff_of_means / np.sqrt((group1_std**2 + group2_std**2) / 2))    
    
    # RUN
    #trace = pm.sample(2000, cores=2)  #  Nota Bene: https://github.com/pymc-devs/pymc3/issues/3388
    trace = pm.sample(1000, tune=1000, cores=1)

pm.kdeplot(np.random.exponential(30, size=10000), shade=0.5);

pm.plot_posterior(trace, varnames=['group1_mean','group2_mean', 'group1_std', 'group2_std', 'ν_minus_one'],
                  color='#87ceeb')

pm.plot_posterior(trace, varnames=['difference of means','difference of stds', 'effect size'],
                  ref_val=0,
                  color='#87ceeb')
                  
pm.forestplot(trace, varnames=['group1_mean',
                               'group2_mean'])

pm.forestplot(trace, varnames=['group1_std',
                               'group2_std',
                               'ν_minus_one'])

pm.summary(trace,varnames=['difference of means', 'difference of stds', 'effect size'])
