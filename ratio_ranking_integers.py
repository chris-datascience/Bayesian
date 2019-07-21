# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:25:04 2019

@author: erdbrca
"""

"""
    Theme:
        Comparing different decision rules for ranking ratios of integer volumes counts/freq etc.
    
    N.B. Bayesian approach is based on <<BMH>> textbook (C.Davidson Pilon) but more accurate 
    by using actual cumul. beta inverse for point estimate.
"""

from itertools import product
import pandas as pd
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt


def compute_upvote_posterior(upvotes, downvotes, LPV_factor=1.65):
    """
    See sec.4.3.5, pg.117 of <<Bayesian methods for hackers>>.
     -Approximates beta posterior pdf of probability of upvote by a normal dist.
     -Then derives 95% percentile ('upper bound') using default LPV_factor    
    N.B. Inputs are integers!
    """
    # --step1: compute mu, std of Normal approx. of posterior--
    a = upvotes + 1
    b = downvotes + 1    
    mu = a / (a + b)
    std_err = np.sqrt((a * b) / ((a + b)**2 * (a + b + 1)))  # standard mu,var from Beta dist. formula.
    # --step2: compute lower & upper bound ('95% least/most plausible value')--
    lb = mu - LPV_factor*std_err
    return lb

def compute_up_beta(u, d, lower_percentile=.05):
    rv = beta(u + 1, d + 1)
    return rv.ppf(lower_percentile)
    #return beta.ppf(lower_percentile, a + 1, b + 1)  # Hmm why different from rv?
    
def get_lower_bound(df_row):
    return compute_up_beta(df_row.Good, df_row.Bad)
    #return compute_upvote_posterior(df_row.Good, df_row.Bad, 2.)

def get_upper_bound(df_row):
    return compute_up_beta(df_row.Good, df_row.Bad, .95)
    #return compute_upvote_posterior(df_row.Good, df_row.Bad, -2.)

def compute_C_score(df, distance_factor=1.):
    df['r'] = np.sqrt(df['Good']**2 + df['Bad']**2)
    df['theta'] = np.arctan(df['Good'] / df['Bad'])
    df['CN_score'] = (df['r']**distance_factor) * np.cos(2*df['theta'])

def compute_C_score_cont_grid(A, B, distance_factor=1.):
    r = np.sqrt(A**2 + B**2)
    theta = np.arctan(B / A)
    return (r**distance_factor) * np.cos(2*theta)

def C_score_compact(A, B, distance_factor=1.):
    r2 = A**2 + B**2
    x = B / A
    return (r2**(distance_factor/2.)) * ((1 - x**2) / (1 + x**2))

    
if __name__=='__main__':

    # [0] Create dataset
    # =============================================================================
    mode = 'all'  # 'random'
    n = 4
    N =  n**2
    min_good, max_good = 1, n
    min_bad, max_mad = 1, n
    df = pd.DataFrame(columns=['Good', 'Bad'], index=range(N))
    df.index.name = 'id'
    if mode=='random':
        df['Good'] = np.random.randint(min_good, max_good, size=N)
        df['Bad'] = np.random.randint(min_bad, max_mad, size=N)
    elif mode=='all':
        P = list(product(range(min_good, max_good+1), repeat=2))
        df['Good'] = [p for p,_ in P]
        df['Bad'] = [p for _,p in P]

    # [1] Upvotes/Downvotes rating using binomial-beta (h.l. without opposite sign correction!)
    # ==========================================================================================
    # Ratio least plausibly 'Bad'
    df['ratio_bad'] = df['Bad'] / (df['Good'] + 0)#1e-6)
    df['ratio_rank'] = df['ratio_bad'].rank(method='min', ascending=True).astype(int)
    # Now running algo and perform ranking based on least_plausible_value:
    df['lpv'] = df.apply(get_lower_bound, axis=1)  # LPV acting as rating (upon which rank is based)
    df['lpv_rank'] = df['lpv'].rank(method='min', ascending=False).astype(int)   # assign ranks low (worst mid) to high (best mid)
    df = df.sort_values(by='lpv_rank', ascending=True)    #  sort according to rank

    # [2] CN approach
    # =============================================================================
    df_final = df.copy()
    compute_C_score(df_final)
    df_final['CN_rank'] = df_final['CN_score'].rank(method='min', ascending=True).astype(int)
    
    # [3] PLOTTING
    # =============================================================================
    # Naive ratio-based rank versus Beta ranks for LPV:
    ax = df_final.plot.scatter(y='ratio_bad', x='lpv_rank', color='DarkBlue', alpha=.5, label='LPV')
    df_final.plot.scatter(y='ratio_bad', x='ratio_rank', color='Red', alpha=.5, label='naive ratio', ax=ax)
    df_final.plot.scatter(y='ratio_bad', x='CN_rank', color='Green', alpha=.5, label='CN ratio', ax=ax)
    plt.title('Ranking differences', fontsize=14)

    print(df_final[['Good','Bad','ratio_rank','lpv_rank','CN_rank']].sort_values('CN_rank', ascending=True))

    # Plot rankings
    df_final.sort_values('ratio_rank')[['Good','Bad']].plot.bar(figsize=(7,4), stacked=True, width=.75)
    plt.ylabel('count', fontsize=13)
    plt.title('ranking based on ratio', fontsize=15)
    
    df_final.sort_values('lpv_rank')[['Good','Bad']].plot.bar(figsize=(7,4), stacked=True, width=.75)
    plt.ylabel('count', fontsize=13)
    plt.title('Bayesian ranking using 95% LPV', fontsize=15)
    
    df_final.sort_values('CN_rank')[['Good','Bad']].plot.bar(figsize=(7,4), stacked=True, width=.75)
    plt.ylabel('count', fontsize=13)
    plt.title('ranking based on CN score', fontsize=15)
    
    #  [4] Measure differences
    # =============================================================================
    CN_vs_ratio = df_final[['ratio_rank','CN_rank']].sort_values('ratio_rank').copy()
    CN_vs_ratio['diff'] = CN_vs_ratio['CN_rank'] - CN_vs_ratio['ratio_rank']
    #CN_diff_mean = CN_vs_ratio['diff'].mean()
    #CN_diff_std = CN_vs_ratio['diff'].std()
    print('\nCN versus ratio rank: ', CN_vs_ratio['diff'].sum())
    print('CN max: ', CN_vs_ratio['diff'].max())
    print('CN min: ', CN_vs_ratio['diff'].min())

    Bayes_vs_ratio = df_final[['ratio_rank','lpv_rank']].sort_values('lpv_rank').copy()
    Bayes_vs_ratio['diff'] = Bayes_vs_ratio['lpv_rank'] - Bayes_vs_ratio['ratio_rank']
    print('\nBayes versus ratio rank: ', Bayes_vs_ratio['diff'].sum())
    print('Bayes max: ', Bayes_vs_ratio['diff'].max())
    print('Bayes min: ', Bayes_vs_ratio['diff'].min())
    