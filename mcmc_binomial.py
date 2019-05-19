# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 13:16:42 2018

@author: Kris
"""

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from scipy import stats
import matplotlib.pyplot as plt
print('Running on PyMC3 v{}'.format(pm.__version__))

"""
<<Bayesian method for hackers>> textbook
PyMC3 implementation of chapter2: Bayesian Logistic Regression:
    https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter2_MorePyMC/Ch2_MorePyMC_PyMC3.ipynb
Also:
    https://docs.pymc.io/notebooks/api_quickstart.html
"""


# =============================================================================
# BASIC BINOMIAL EXAMPLE in SECTION 2.2.X:
# =============================================================================


N = 100; X = 15
p_expected = X / N
with pm.Model() as model:
    #p = pm.Uniform("true_chargeback_rate", 0, 1, testval=p_expected)
    #p = pm.Normal("true_rate", mu=p_expected, sd = .01, testval=p_expected) # doesnt work, might give negative probability
    p = pm.Beta("true_rate", alpha=2, beta=2)

# Add observations:
with model:
    observations = pm.Binomial("observed_chargeback_volume", N, p, observed=X)

# Run simulation & sample:
with model:
    #step = pm.Metropolis(vars=[p])
    #trace = pm.sample(40000, step=step) # overrides auto-assigning sampler
    start = pm.find_MAP()
    trace = pm.sample(40000, cores=1) # no sampler speficied so auto-picks one
    burned_trace = trace[15000:]

# --Plotting results---
plt.figure(figsize=(12, 5))
p_trace = burned_trace["true_rate"][15000:]
#p_trace = burned_trace["freq_cheating"][3000:]
plt.hist(p_trace, histtype="stepfilled", density=True, alpha=0.85, bins=30, \
         label="posterior distribution", color="#348ABD")
#plt.vlines([.05, .35], [0, 0], [5, 5], alpha=0.3)
plt.vlines([p_expected]*2, [0]*2, [12]*2, 'k', alpha=0.5)
plt.xlim(0, 1)
plt.legend()

print('\nExpected value: %2.3f' % (p_expected))
print('\nMean posterior value: %2.3f' % np.mean(p_trace))
print('5%% percentile: %2.3f' % np.percentile(p_trace, 5))
print('95%% percentile: %2.3f' % np.percentile(p_trace, 95))

# ---------------------------------------------------------------------------------------
#with pm.Model() as model:
#    parameter = pm.Exponential("poisson_param", 1.0)
#    #parameter = pm.Exponential("poisson_param", 1.0, testval=0.5)  # specify starting point
#    data_generator = pm.Poisson("data_generator", parameter)
#    betas = pm.Uniform("betas", 0, 1, shape=3)  # for multivariable problems, i.e. variable size (3,)

#parameter.tag.test_value
"""
Two ways to create deterministic variables in PyMC3:
 (1) Elementary operations, like addition, exponentials etc. implicitly create deterministic variables.
 (2) call and define function as second argument: deterministic_variable = pm.Deterministic("deterministic variable", some_function_of_variables)
"""
#detvar = pm.Deterministic("deterministic variable example", data_generator + 2)
#colors = ["#348ABD", "#A60628"]
