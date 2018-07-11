import numpy as np
from scipy import stats,pi,sqrt,exp,linspace, optimize, interpolate
from scipy.special import erf
from sklearn.decomposition import FastICA, PCA
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import normalize

########### ########### ########### ########### ########### ###########
# Functions
########### ########### ########### ########### ########### ###########
def amari_error(L0,Lhat):
	''' Amari Error: Measure of how close W := inv(B) * A  is to identity matrix
	Invariant with column ordering & scaling
	'''
	W = np.matmul(np.linalg.inv(L0), Lhat)
	m = W.shape[0]
	r0 = np.sum(np.multiply(np.sum(abs(W),0), 1/np.amax(abs(W),0))) - m
	r0 += np.sum(np.multiply(np.sum(abs(W),1), 1/np.amax(abs(W),1))) - m
	r0 = r0/(2*m*(m-1))
	return r0

def map_mixing_matrix_to_paramSpace(E0):
	'''
	Map nonsingular (m x m)-dimensional matrix to parameter space of Mixing Matrices
	Ensures Identifiability with respect to column permutations & scaling.
	'''
	# Normalize columns by Euclidean Normal
	d1 = np.diagflat(1/np.sqrt(np.sum(np.multiply(E0,E0),0)))
	E1 = np.matmul(E0,d1)
	m = E1.shape[1]
	# Permute Columns
	for j in range(0,m-1):
		h1 = abs(E1[j,j:m])
		jmax = np.argmax(h1) + j
		E1[:,[j,jmax]] = E1[:,[jmax,j]]
	# Standardize so diagonals equal 1
	E2 =  np.matmul(E1, np.diagflat(1/np.diag(E1)))
	return E2

########### ########### ########### ########### ########### ###########
# REstimator Class
########### ########### ########### ########### ########### ###########
class REstimator():
	def __init__(self, X, steps = 1, PrelimEst = None, type_REst = "R", type_score = "Kernel"):
		'''
		Initialize Class Object with Inputs
		Required: X  --> (n x m)-dim matrix (with rows indexing Time or observation)
		Optional: 
		-> PrelimEst (If no preliminary estimator provided, FastICA is used as default.)
		-> steps (# of times to iterate one-step R-estimation procedure)
		-> type_score (# Use scores from skew-normal family (type_score = "Parametric".
		             Default type_score relies on scores from Gaussian-kernel density estimates)
		'''
		self.X = X
		self.type_REst = type_REst
		self.type_score = type_score
		self.steps = steps

		if PrelimEst is None:
			self.E0_ = get_prelim_est(X)
		else:
			self.E0_ = PrelimEst

	# Get REstimator
	def get_REstimator(self):
		N, m = self.X.shape[0], self.X.shape[1]
		# Map to Preliminary to Parameter Space
		self.L0_ = map_mixing_matrix_to_paramSpace(self.E0_)
		# Append to Multi-step Estimators List
		self.estimators = [self.L0_]
		self.e = []
		# Run Multi-step R-Estimator
		for j in range(self.steps):
			print('Preliminary Estimator:')
			print(self.L0_)
			# Get Residuals
			self.Z0_ = np.matmul( np.linalg.inv(self.L0_), self.X.transpose()).transpose()
			# Get Maximal Invariant under Nuisance
			self.maximal_invariant()
			# Estimate Scores on Grid
			self.estimate_score_function_on_Grid()
			# Interpolate Rank-based Scores 
			self.interpolate_rank_scores()
			# Estimate Cross-Information Quantities
			Nhat = self.estimate_CIQ()
			Update = np.matmul(self.L0_, Nhat - np.diag(np.diag(np.matmul(self.L0_, Nhat))))
			e = self.L0_ + N**(-1/2) * Update
			print(e - self.L0_)
			self.L0_ = map_mixing_matrix_to_paramSpace(e)
			print('R-Estimator:')
			print(self.L0_)
			self.estimators.append(self.L0_)


	def maximal_invariant(self):
		N,m = self.Z0_.shape	
		if self.type_REst == "R":
			self.MI = np.empty(self.Z0_.shape)
			for j in range(self.Z0_.shape[1]):
				self.MI[:,j] = stats.rankdata(self.Z0_[:,j])/(N+1)
		

	# Estimate Score Function
	def estimate_score_function_on_Grid(self):
		''' 
		Scores for each component of self.Z0_, estimated over a Grid 
		'''
		self.Scores_Grid = []
		m = self.Z0_.shape[1]
		for j in range(0,m):
			zj = self.Z0_[:,j]
			if self.type_score == 'Parametric' and self.type_REst == "R":
				sj = get_skewt_scores_on_grid(zj)
			elif self.type_score == 'Parametric' and self.type_REst == "SR":
				sj = get_t_scores()
			else:
				sj = get_1D_kernel_on_grid(zj)
			self.Scores_Grid.append(sj)


	# Interpolate Rank Scores
	def interpolate_rank_scores(self, R = None):
		if R is None:
			R = self.MI
			update_rank_scores = True
		else:
			update_rank_scores = False
		rank_icdf, rank_phi = np.empty(self.Z0_.shape), np.empty(self.Z0_.shape)
		for j in range(len(self.Scores_Grid)):
			Uj = self.Scores_Grid[j]
			t0, cdf, phi = Uj['t0'], Uj['cdf'], Uj['phi']
			interp_cdf = interpolate.interp1d(cdf[:,0], t0[:,0])
			interp_phi = interpolate.interp1d(t0[:,0], phi[:,0])
			icdf = interp_cdf(R[:,j])
			rank_icdf[:,j] = icdf
			rank_phi[:,j] = interp_phi(icdf)
		N, m = self.Z0_.shape[0], self.Z0_.shape[1]
		T0 = np.zeros(self.E0_.shape)
		for i in range(0,N):
			T0 = T0 + np.matmul(rank_phi[i,:].reshape(m,1), rank_icdf[i,:].reshape(1,m))
		T0 = T0/N
		mean_phi = np.mean(rank_phi, axis=0).reshape(m,1)
		mean_icdf =  np.mean(rank_icdf, axis=0).reshape(1,m)
		T0 = np.sqrt(N) * (T0 - np.matmul(mean_phi, mean_icdf))
		Scores_Rank = T0 - np.diag(np.diag(T0))
		if update_rank_scores is True:
			self.Scores_Rank = Scores_Rank
		return T0

	# Estimate Cross-Information Quantities
	def estimate_CIQ(self):
		N, m = self.Z0_.shape[0], self.Z0_.shape[1]
		trsL = []
		for r in range(m):
			for s in range(m):
				if r != s:
					trsL = trsL + [{'type':1, 'r':r, 's':s}]
					trsL = trsL + [{'type':2, 'r':r, 's':s}]
		gamma, rho = np.zeros([m,m]), np.zeros([m,m])
		for j in range(0,len(trsL)):
			trs = self.optimize_CIQ(trsL[j])
			r,s,ciq = trs['r'], trs['s'], trs['CIQ']
			if trs['type'] == 1:
				gamma[r,s] = ciq
			else:
				rho[r,s] = ciq
		A, B, Denom = np.zeros([m,m]), np.zeros([m,m]), np.zeros([m,m])
		for r in range(m):
			for s in range(m):
				if r != s:
					Denom[r,s] = gamma[r,s] * gamma[s,r] - rho[r,s] * rho[s,r]
					if Denom[r,s] != 0:
						A[r,s] = gamma[r,s] / Denom[r,s]
						B[r,s] = -rho[r,s] / Denom[r,s]
		S = self.Scores_Rank
		Nhat = np.matmul(A.transpose(), S) + np.matmul(B.transpose(), S.transpose())
		return Nhat

	def optimize_CIQ(self, trs):
		N, m, T0 = self.X.shape[0], self.X.shape[1], self.Scores_Rank
		grid = np.concatenate((linspace(0,0.1,10),linspace(0.2,1,9),linspace(2,10,9)))
		flag = False
		uj = self.CIQ_objective(grid[0], trs)
		u0 = [uj]
		for j in range(1,len(grid)):
			uj = self.CIQ_objective(grid[j], trs)
			if np.sign(uj)*np.sign(u0[j-1]) < 0:
				flag = True
				break;
			else:
				u0.append(uj)

		if flag is True:
			y1,y2 = u0[len(u0)-1], uj
			x1,x2 = grid[j-1], grid[j]
			slope = (y2 - y1)/(x2 - x1)
			inter = y1 - slope*x1
			ciqi = -inter/slope
			Hrc = np.sign(ciqi)*min(abs(1/ciqi),100)
		else:
			Hrc = 0.1
		trs['CIQ'] = Hrc
		return trs

	# Objective function: CIQ
	def CIQ_objective(self, l, trs):
		X, L0_, T0 = self.X, self.L0_, self.Scores_Rank 
		N, m = X.shape[0], X.shape[1]
		er, es = np.zeros([m,1]), np.zeros([m,1])
		er[trs['r']], es[trs['s']]  = 1, 1
		ers = np.matmul(er, es.transpose())
		u = ers - np.diag(np.diag(np.matmul(L0_, ers)))
		Lx1 = 1/sqrt(N) * np.matmul(L0_ ,u)

		if trs['type'] == 1:
			r1,s1 = trs['r'], trs['s']
		else:
			r1,s1 = trs['s'], trs['r']

		T0rs = T0[r1,s1]
		Ex2 = L0_ + l * T0rs * Lx1
		Lx2 = map_mixing_matrix_to_paramSpace(Ex2)
		Diff = abs(Ex2 - Lx2)
		#if abs(Diff.max()) > 0.0001:
		#	print('error')
		Zx2 =  np.matmul(np.linalg.inv(Lx2), X.transpose()).transpose() 
		Rx2 =  np.empty(X.shape)
		for j in range(X.shape[1]):
			Rx2[:,j] = stats.rankdata(Zx2[:,j])/(N+1)
		Tlrs = self.interpolate_rank_scores(R = Rx2)
		return T0[r1,s1] * Tlrs[r1,s1]


########### ########### ########### ########### ########### ###########
# Other Functions
########### ########### ########### ########### ########### ###########
def get_prelim_est(X, type_prelim = 'FastICA'):
	m = X.shape[1]
	if type_prelim == 'FastICA':
		fica = FastICA(n_components=m, w_init=np.identity(m), random_state = 100)
		a = fica.fit_transform(X)
		E_ = fica.mixing_
	else:
		print("valid preliminary estimator no specified; using FastICA")
		E_ = get_prelim_est(X)
	return E_

def get_performance(L0, LHat):
	y = []
	for j in range(len(LHat)):
		yj = amari_error(L0, LHat[j])
		y.append(yj)
	return y


def get_1D_kernel_on_grid(xj,epsilon = 10**(-6)):
	xj = xj.reshape(-1, 1)
	xj_sd = np.std(xj)
	# Rule-of-thumb bandwidth estimator
	bw = 1.06*xj_sd*len(xj)**(-0.2)
	# Kernel Density Estimator
	kj = KernelDensity(kernel='gaussian', bandwidth=bw).fit(xj)
	# Output List
	U = {}
	# Grid (t0)
	npoints = max(np.floor(len(xj)*1.1), 5000)
	t0 = linspace(min(xj)-np.std(xj)/2,max(xj)+np.std(xj)/2, npoints).reshape(-1,1)
	dt = t0[1] - t0[0]
	epsilon = min(epsilon,dt/4)
	# f: Location Score (phi)
	f = lambda t: np.exp(kj.score_samples(t))
	f0 = f(t0)
	fprime = (f(t0+epsilon/2) - f(t0-epsilon/2))/epsilon
	fprime[f0 < 10**(-15)] = 0
	phi = -fprime/f0
	# CDF
	cdf = f0.cumsum()*dt
	U = {'t0':t0, 'phi' : phi.reshape(-1,1), 'cdf' : cdf.reshape(-1,1)}
	return U

def get_skewt_scores_on_grid(xj,epsilon = 10**(-6)):
	xj = xj.reshape(-1, 1)
	# Output List
	U = {}
	# Grid (t0)
	npoints = max(np.floor(len(xj)*1.1), 5000)
	t0 = linspace(min(xj)-np.std(xj),max(xj)+np.std(xj), npoints).reshape(-1,1)
	dt = t0[1] - t0[0]
	epsilon = min(epsilon,dt/4)
	# f: Location Score (phi)
	fitp = stats.skewnorm.fit(xj)
	thetaHat = [fitp[1], fitp[2], fitp[0]]
	f = lambda t: stats.skewnorm.pdf(t, a = fitp[0], loc = fitp[1], scale = fitp[2])
	f0 = f(t0)
	fprime = (f(t0+epsilon/2) - f(t0-epsilon/2))/epsilon
	fprime[f0 < 10**(-15)] = 0
	phi = -fprime/f0
	# CDF
	cdf = stats.skewnorm.cdf(t0, a = fitp[0], loc = fitp[1], scale = fitp[2])
	U = {'t0':t0, 'phi' : phi.reshape(-1,1), 'cdf' : cdf.reshape(-1,1)}
	return U



########### ########### ########### ########### ########### ###########
'''
class Skew_t:
	def __init__(self,x,location=0.,scale=1.,shape=0.,dof=10):
		self.x = x
		if abs(shape) > 15:
			shape = np.sign(shape)*15
		self.theta = [location, scale, shape,dof]
		u = self.get_mle_sn()
		self.theta[0:3] = [u[1],u[2],u[0]]

	def print_theta(self):
		print("Location: " + str(self.theta[0]))
		print("Scale: " + str(self.theta[1]))
		print("Shape: " + str(self.theta[2]))
		print("DOF: " + str(self.theta[3]))

	def get_mle_sn(self):
		mu,sigma,shape,dof = self.theta[0], self.theta[1], self.theta[2],self.theta[3]
		self.Fit1 = stats.skewnorm.fit(self.x)
		return self.Fit1

	def interpolate_rank_scores(self, epsilon = 0.001):
		# Interpolate
		n = 2**10
		G = linspace(min(self.x),min(self.x),n)
		
		mu,sigma,shape,dof = self.theta[0], self.theta[1], self.theta[2],self.theta[3]
		dneg = stats.skewnorm.pdf(G - epsilon/2, a = shape, loc = mu, scale = sigma)
		dpos = stats.skewnorm.pdf(self.x + epsilon/2, a = shape, loc = mu, scale = sigma)
		self.fprime = (dpos - dneg)/epsilon
		return self.fprime

	def skewnorm_score(self, epsilon = 0.001):
		mu,sigma,shape,dof = self.theta[0], self.theta[1], self.theta[2],self.theta[3]
		dneg = stats.skewnorm.pdf(self.x - epsilon/2, a = shape, loc = mu, scale = sigma)
		dpos = stats.skewnorm.pdf(self.x + epsilon/2, a = shape, loc = mu, scale = sigma)
		self.fprime = (dpos - dneg)/epsilon
		return self.fprime

	def pdf(self):
		mu,sigma,shape,dof = self.theta[0], self.theta[1], self.theta[2],self.theta[3]
		self.density = stats.skewnorm.pdf(self.x, a = shape, loc = mu, scale = sigma)
		return self.density
		#z = (self.x - mu) / sigma
		# If Skew-T
		#z1 = ((dof+1)/(dof+z*z))**0.5
		#self.density = (2/sigma) * stats.t.cdf(z,dof) *  stats.t.cdf(shape*z*z1,dof+1)
		#self.density = 2 / sigma * stats.norm.pdf(z) * stats.norm.cdf(shape*z)

	def negloglikelihood(self, theta = None):
		if theta is not None:
			self.theta = theta
		self.L = -sum(np.log(self.pdf()))
		return self.L

	def mle(self):
		con1 = ({'type': 'ineq', 'fun': lambda x: x[1] - 0.1 }, \
			{'type': 'ineq', 'fun': lambda x: x[3] - 1 }, \
			{'type': 'ineq', 'fun': lambda x: x[2] + 15 },\
			{'type': 'ineq', 'fun': lambda x: -x[2] + 15 })
		b = optimize.minimize(self.negloglikelihood,self.theta,constraints=con1)
		self.theta = b.x
		return b
'''

