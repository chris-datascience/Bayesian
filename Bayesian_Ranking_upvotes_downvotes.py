# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 20:47:32 2018

@author: Kris
"""

import pandas as pd
import numpy as np

        
def compute_upvote_posterior(upvotes, downvotes):
    """
    See sec.4.3.5, pg.117 of <<Bayesian methods for hackers.
    -Computes posterior pdf of probability of upvote (approximated by a normal dist.)
    -Then derives 95% percentile ('upper bound')
    
    N.B. Inputs are integers!
    """
    # --step1: compute mu, std of Normal approx. of posterior--
    a = upvotes + 1
    b = downvotes + 1    
    mu = a / (a + b)
    std_err = np.sqrt((a * b) / ((a + b)**2 * (a + b + 1)))  #  inference magic
    # --step2: compute lower & upper bound ('95% least/most plausible value')--
    lb = mu - 1.65*std_err
    return lb

def compute_downvote_posterior(upvotes, downvotes):
    """
    See sec.4.3.5, pg.117 of <<Bayesian methods for hackers>> texbook.
    -Computes posterior pdf of probability of downvote (beta approximated by a normal dist.)
    -Then derives 5% percentile, called '95% least plausible value' aka. lower bound.
    -Inputs should be integers.
    """
    # --step1: compute mu, std of Normal approx. of posterior--
    a = downvotes + 1
    b = upvotes + 1
    mu = a / (a + b)
    std_err = np.sqrt((a * b) / ((a + b)**2 * (a + b + 1)))  #  inference magic
    # --step2: compute lower bound ('95% least plausible value')--
    lb = mu - 1.65*std_err
    return lb

def get_lower_bound(df_row):
    return compute_downvote_posterior(df_row.Good_volume, df_row.Bad_volume)
    
  
if __name__=='__main__':
    
    # APPLICATION EXAMPLE
    # ===================
    # Making up some data:
    df = pd.DataFrame(columns=['Good_volume', 'Bad_volume'], \
                      index=np.random.randint(1000, 8000, size=100))
    df.index.name = 'id'
    df['Good_volume'] = np.random.randint(10, 500, size=100)
    df['Bad_volume'] = (np.random.uniform(0, 1, size=100) * df['Sales_volume']).astype(int)
    df['ratio_bad'] = df['Bad_volume'] / df['Good_volume']  # added for comparison
    df['Bad_ratio'] = np.round(df['Bad_ratio'], 3)
    
    # Now running algo and perform ranking
    df['least_plausible_value'] = df.apply(get_lower_bound, axis=1)
    df['least_plausible_value'] = np.round(df['least_plausible_value'], 3)
    df = df.sort_values(by='least_plausible_value', ascending=False)    #  sort according to upper_bound
    df['rank'] = [r+1 for r in range(len(df))]   # assign ranks low (worst mid) to high (best mid)
    print(df[:20])
    
    # Testing:
#    df['total_vol'] = df['Sales_volume'] + df['CB_volume']
#    print(df.iloc[np.argmax(posterior_mean)])    
#    print(df.iloc[np.argmin(posterior_mean)])
#
#    print(df.iloc[np.argmax(posterior_std)])
#    print(df.loc[df['total_vol'].idxmin(),:])
#    
#    print(df.iloc[np.argmin(posterior_std)])
#    print(df.loc[df['total_vol'].idxmax(),:])

    # -- COMPUTATION OUTSIDE OF DATAFRAME: --
    #posterior_mean, posterior_std = compute_upvote_posterior(df.Sales_volume.values, df.CB_volume.values)
    #lb = lower_bound(posterior_mean, posterior_std)  #  Compute lower_bound for each merchant
    # .. or alternatively use order = np.argsort(-upper_bound) to get indexed ranking directly
