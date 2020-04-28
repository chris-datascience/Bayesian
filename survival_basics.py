# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:51:29 2020

@author: Kris
"""

# =============================================================================
# Monitor exponential delays in one-event survival data over time
# =============================================================================

#import os, sys
#from random import sample, choice
#from collections import namedtuple
import numpy as np
import pandas as pd
#from scipy.integrate import quad
#from scipy.stats import expon, lomax, gamma
from lifelines import KaplanMeierFitter
from lifelines.datasets import load_waltons
from lifelines.utils import datetimes_to_durations, median_survival_times
import matplotlib.pyplot as plt
#from exponential_post_predictive import Bayesian_conjugate_inference
#from decay_problem_MacKay import expon_integral

plt.style.use('bmh')


def manual_KaplanMeier():
    pass


if __name__=='__main__':
    df = load_waltons()
    dfC = df[df.group=='control'].copy()
    
    T = dfC['T']
    E = dfC['E']
    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=E)
    
    #kmf.survival_function_
    #kmf.cumulative_density_
#    kmf.plot_survival_function()
    
    # COMPUTE MANUAL K-M Survivor estimates
    del dfC['group']
    dfC = dfC.sort_values('T')
    dfC = dfC.astype(int)#set_index('T')
    
    #dfC_KM = dfC.groupby('T').sum()
    dfC_KM = pd.concat([pd.DataFrame(data=np.zeros((1,2)), columns=list('TE')), dfC.copy()], axis=0)
    dfC_KM.index = range(len(dfC_KM))
    total_events = dfC.E.sum() #len(dfC)
    total_potential = len(dfC)
    dfC_KM['n_at_risk'] = total_potential - dfC_KM['E'].cumsum()
    dfC_KM['n minus d'] = dfC_KM['n_at_risk'] - dfC_KM['E']
    dfC_KM['KMfactor'] = dfC_KM['n minus d'] / dfC_KM['n_at_risk']
    dfC_KM['S_est'] = dfC_KM['KMfactor'].cumprod()
    SurvEst = dfC_KM[['T','S_est']].copy()
    SurvEst = SurvEst.drop_duplicates(subset='T', keep='last')
    SurvEst['dt'] = SurvEst['T'].diff().fillna(0)
    SurvEst['hazard'] = (SurvEst['S_est'].shift(1) - SurvEst['S_est']) / (SurvEst['dt'] * SurvEst['S_est'])
    SurvEst['cumhazard'] = SurvEst['hazard'].cumsum()
    #dfC_KM['survived'] = len(dfC) - dfC_KM['E']  #total_events - dfC_KM['E']
    #dfC_KM['KMfactor'] = 1 - dfC_KM['E'] / dfC_KM['survived']
#    S_est = np.empty((len(dfC_KM)+1,))
#    S_est[0] = 1
#    for i,val in enumerate(dfC_KM.KMfactor.values):
#        S_est[i+1] = S_est[i] * val
    
    # Inspect result: compare to lifelines values
    S_est = SurvEst['S_est'].values
    print(np.allclose(kmf.survival_function_.KM_estimate.values, S_est))
    t_KM = np.hstack((0,dfC_KM.index.values))
    #plt.figure(figsize=(7,6))
    #plt.plot(t_KM, kmf.survival_function_.KM_estimate.values, 'm-')
    kmf.plot_survival_function(figsize=(7,6))
    #plt.plot(dfC_KM['T'].values, S_est, 'r.', alpha=.5, label='manual')
    plt.plot(SurvEst['T'].values, SurvEst['S_est'].values, 'r.:', alpha=.6, label='manual')
    plt.legend()
    
    kmf.cumulative_density_.plot(figsize=(7,6))
    plt.plot(SurvEst['T'].values, (SurvEst['hazard']*SurvEst['S_est']).cumsum().values, 'r.:', alpha=.6, label='manual')
    plt.legend()
