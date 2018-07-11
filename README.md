# ICA-R-Estimation

Reference:

M. Hallin & C. Mehta (2015). R-estimation for asymmetric independent component analysis. Journal of the American Statistical Association, 110(509), 218-232 

Independent component analysis (ICA) is an approach to multivariate statistics wherein observed signals are deconvolved, or separated, into independent latent source signals. In the ICA model, observed m-vectors <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{X}(t)&space;:=&space;[X_1(t),X_2(t),\ldots,X_m(t)]^{\prime}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{X}(t)&space;=&space;[X_1(t),X_2(t),\ldots,X_m(t) satisfy 
  
 <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{X}(t)&space;=&space;\mathbf{\Lambda}&space;\mathbf{S}(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{X}(t)&space;=&space;\mathbf{\Lambda}&space;\mathbf{S}(t)" title="\mathbf{X}(t) = \mathbf{\Lambda} \mathbf{S}(t)" /></a>, 
 
where <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{\Lambda}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{\Lambda}" title="\mathbf{\Lambda}" /></a> is a nonsingular <a href="https://www.codecogs.com/eqnedit.php?latex=(m&space;\times&space;m)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(m&space;\times&space;m)" title="(m \times m)" /></a>-dimensional mixing matrix and 

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{S}(t)&space;:=&space;[S_1(t),S_2(t),\ldots,S_m(t)]^{\prime}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{S}(t)&space;:=&space;[S_1(t),S_2(t),\ldots,S_m(t)]^{\prime}" title="\mathbf{S}(t) := [S_1(t),S_2(t),\ldots,S_m(t)]^{\prime}" /></a> is a vector whose components S_k(t) have pairwise independent distributions (over t=1,2,...). 

A primary objective of ICA is to estimate the mixing matrix (<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{\Lambda}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{\Lambda}" title="\mathbf{\Lambda}" /></a>) from the observed X-vectors. Applying the inverse of an accurate estimate of the mixing matrix to the observed mixed X-vectors allows for recovery of the source signals in the ICA model. 

In this project, we propose a one-step R-estimator for the mixing matrix that aims to achieve greater robustness against source signals with heavy-tailed distributions and other types of noise relative to extant estimators for the mixing matrix. Moreover, we are able to spell out the R-estimator's asymptotic properties, such as its limiting distribution, through a semi-parametric procedure.

Evaluating R-estimator requires first

1. obtaining a preliminary estimator L0 of the mixing matrix that achieves root-n consistency and
2. specifying a m-tuple of univariate distributions f:=(f1,...,fm) for the respective unobserved independent source signals.

Here, we can take our preliminary estimator (L0) to be from any method, including FastICA (R: fastICA; Python: scikit-learn/Decomposition) and kernelICA (Matlab: https://www.di.ens.fr/~fbach/kernel-ica/kernel-ica1_2.tar.gz), that have strong empirical performances.

In this code, we estimate the distributions, f:=(f1,...,fm), needed for evaluating the R-estimator by fitting either (a) kernel density estimates or (b) a family of parametric distributions to each of the m unobserved source signals under the preliminary estimator. 

The one-step R-estimator updates the preliminary estimator with a term that depends on f-scores for a parametric verision of the ICA model. Unlike parametric methods, however, the f-scores are not evaluated at residuals (source signals inferred by the preliminary estimator). The f-scores are instead evaluated at the inverse of the empirical f-distribution function for each residual component, which depends on the componentwise ranks of each residual. 

### Using the code
The file REst_Func.py defines a class called REstimator. The class is initialized with choice of Preliminary Estimator (if none provided, fastICA is used) and the type of f-score estimation: Gaussian Kernel (default) or parametric skew-normal. Then the class method get_REstimator() obtains the estimator. The R-estimation procedure can be iterated multiple times by using the R-estimator from the prior iteration as the preliminary estimate.

The Class object "estimators" will be a list of Q+1 estimators (where Q=#steps used for estimation). Amari error is a popular metric for evaluating performance of mixing matrix estimators in simulation experiments, where a ground truth is known. The function "get_performance" in the REst_Func.py file will evaluate the Amari error between the ground truth and a list of estimators.

```python
import REst_Func as REst
import numpy as np
from scipy import stats
from sklearn.decomposition import FastICA

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
# -> steps (# of times to iterate one-step R-estimation procedure, with step 0 = PrelimEst)
# -> type_score (# Use scores from skew-normal family (type_score = "Parametric".
#              Default type_score relies on scores from Gaussian-kernel density estimates)
M = REst.REstimator(X,PrelimEst=E0_, steps = 2, type_score="Parametric")
# Run R-Estimation
M.get_REstimator()

# Check Amari error of n-step R-Estimators
# Step 0 = Preliminary Estimate
REst.get_performance(L, M.estimators)

```
