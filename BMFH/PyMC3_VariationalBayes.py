# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:41:37 2019

@author: erdbrca
"""

import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
#import theano
import numpy as np

"""
    Source:
        https://docs.pymc.io/notebooks/variational_api_quickstart.html
"""
     
w = pm.floatX([.2, .8])
mu = pm.floatX([-.3, .5])
sd = pm.floatX([.1, .1])

with pm.Model() as model:
    x = pm.NormalMixture('x', w=w, mu=mu, sd=sd)#, dtype=theano.config.floatX)
    x2 = x ** 2
    sin_x = pm.math.sin(x)
    
    pm.Deterministic('x2', x2)
    pm.Deterministic('sin_x', sin_x)
    
    # Run No-U-Turn sampler
    trace = pm.sample(50000)
    
# Inspect results
pm.traceplot(trace)

# Same model, with ADVI this time
with pm.Model() as model:
    x = pm.NormalMixture('x', w=w, mu=mu, sd=sd) #, dtype=theano.config.floatX)
    x2 = x ** 2
    sin_x = pm.math.sin(x)
    
    # Automatic differentiation variational inference (ADVI).
    mean_field = pm.fit(method='advi')

pm.plot_posterior(mean_field.sample(1000), color='LightSeaGreen')

# Comparison:
ax = sns.kdeplot(trace['x'], label='NUTS');
sns.kdeplot(approx.sample(10000)['x'], label='ADVI