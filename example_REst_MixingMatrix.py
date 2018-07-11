import REst_Func as REst
import numpy as np
from scipy import stats
from sklearn.decomposition import FastICA
#import pandas as pd
#from scipy import signal
#import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
#from scipy import interpolate

########### ########### ########### ########### ########### ###########
# Generate N replicates of m Independent Random Variables with distributions g1,g2,g3
# S(t) = [S1(t),S2(t),...,Sm(t)] 
# S = (N x m)-dimensional matrix with rows S(t) for t=1,...,N.
########### ########### ########### ########### ########### ###########
N = 10000
S = np.c_[stats.laplace.rvs(size = N),
          stats.laplace.rvs(size = N),
          stats.skewnorm.rvs(size = N, a = -4)]

########### ########### ########### ########### ########### ###########
# Mix Data with non-singular (m x m)-dimensional mixing matrix L
# X(t) = L S(t)'
# X = (N x m)-dimensional matrix with rows X(t) for t=1,...,N.
L = np.diag([1.,1.,1.])
L[[0,1],[1,0]] = 0.25
print(L)
X = np.matmul(S,np.transpose(L))


########### ########### ########### ########### ########### ###########
# Get Preliminary Estimate of L from X
# Any root-n consistent estimator is valid.
# FastICA from scikit-learn's Decomposition module used below.
fica = FastICA(n_components=m, w_init=np.identity(m), random_state = 100)
a = fica.fit_transform(X)
E0_ = fica.mixing_


########### ########### ########### ########### ########### ###########
# Run R-Estimation Procedure
# Initialize Class Object with Inputs
# Required: X  --> (n x m)-dim matrix (with rows indexing Time or observation)
# Optional: 
# -> PrelimEst (If no preliminary estimator provided, FastICA is used as default.)
# -> steps (# of times to iterate one-step R-estimation procedure)
# -> type_score (# Use scores from skew-normal family (type_score = "Parametric".
#              Default type_score relies on scores from Gaussian-kernel density estimates)
M = REst.REstimator(X,PrelimEst=E0_, steps = 2, type_score="Parametric")
# Run R-Estimation
M.get_REstimator()

# Check Amari error of n-step R-Estimators
# Step 0 = Preliminary Estimate
REst.get_performance(L, M.estimators)
