# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:32:17 2020

@author: Kris
"""
import time
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(13)
rho = .6  # correlation coeff.
N = 500000  # sample size

# [1] Slow loop for reference
# ---------------------------
t0 = time.time()
X = np.empty(N, dtype=float)
X[0] = np.random.normal(loc=0, scale=1, size=1)
for i in range(1,N):
    #X[i] = rho*X[i-1] + np.random.normal(loc=0, scale=np.sqrt(1 - rho**2), size=1)  # equivalent
    X[i] = np.random.normal(loc=rho*X[i-1], scale=np.sqrt(1 - rho**2), size=1)
print('Stopwatch 1:   %2.3f sec' % (time.time() - t0))
plt.plot(range(N), X, 'b-')
plt.figure()
plt.hist(X, density=True, color='blue', alpha=.2, bins=31)

# [2] Pandas vectorisation Attempt
# ===============================
#np.random.seed(13)
#t0 = time.time()
#df = pd.DataFrame(dtype=float)
#df['X'] = np.random.normal(loc=0, scale=np.sqrt(1 - rho**2), size=N)
#df['X0'] = 0
#df.loc[0,'X0'] = np.random.normal(loc=0, scale=1, size=1)
##df['rho_X'] = rho * df['X'].shift(1)
##df['Y'] = df['X'] + df['rho_X']
#df['Y'] = df['X'].rolling(window=2, min_periods=2).apply(lambda W: np.dot(W, [rho,1.]), raw=False)
#
#print('Stopwatch 2:   %2.3f sec' % (time.time() - t0))
#print(np.allclose(X, df.Y.values))
#

# [3] Accumulate (=reduce but storing intermediate results in a generator (whereas reduce would only return final value))
# -----------------------------------------------------------------------------------------------------------------------
from itertools import accumulate
np.random.seed(13)
t0 = time.time()
L = list(np.random.normal(loc=0, scale=1, size=1))
L.extend(list(np.random.normal(loc=0, scale=np.sqrt(1 - rho**2), size=N-1)))
Z = list(accumulate(L, lambda x,y: rho*x + y))
print('\nStopwatch 2:   %2.3f sec' % (time.time() - t0))
print(np.allclose(X, Z))

#plt.figure()
plt.hist(Z, density=True, color='green', alpha=.2, bins=31)

# [4] Matrix multiplication
# -------------------------
from scipy.linalg import circulant
from scipy.sparse import tril
np.random.seed(13)
N_small = 10000  # WARNING: don't wanna go much higher than this...--> too slow, too much memory!
t0 = time.time()
v = np.hstack((np.random.normal(loc=0, scale=1, size=1), \
               np.random.normal(loc=0, scale=np.sqrt(1 - rho**2), size=N_small-1)))
M = tril(circulant(np.power(rho, np.arange(0,N_small))))#.toarray()
Z_small = M.dot(v)
print('\nStopwatch 4:   %2.3f sec' % (time.time() - t0))
print(np.allclose(X[:N_small], Z_small))

# [5] Scipy filtering
# -------------------
# to do; using lfilt etc.

# =============================================================================
# # TRY MORE AT https://stackoverflow.com/questions/47427603/recurrence-with-numpy
# ---------------------------------------------------------------------------------

# [6] Loop with memoization and Numba JIT
# ---------------------------------------
from numba import jit
np.random.seed(13)
t0 = time.time()
Rnd = np.empty(N, dtype=float)
Rnd[0] = np.random.normal(loc=0, scale=1, size=1)
Rnd[1:] = np.random.normal(loc=0, scale=np.sqrt(1 - rho**2), size=N-1)
@jit(nopython=True)
def loop_recurrence(x):
    n = x.shape[0]
    q = np.empty(n)
    q[0] = x[0]
    for i in range(1,n):
       q[i] = .6 * q[i-1] + x[i]
    return q
ZZ = loop_recurrence(Rnd[:20])
print('\nStopwatch Numba during compilation:   %2.3f sec' % (time.time() - t0))

np.random.seed(13)
t0 = time.time()
Rnd = np.empty(N, dtype=float)
Rnd[0] = np.random.normal(loc=0, scale=1, size=1)
Rnd[1:] = np.random.normal(loc=0, scale=np.sqrt(1 - rho**2), size=N-1)
ZZZ = loop_recurrence(Rnd)
print('Stopwatch Numba COMPILED:   %2.8f sec' % (time.time() - t0))
print(np.allclose(X, ZZZ))

