# Author:		Chintan Mehta. Pleae e-mail cmehta@princeton.edu AND chintanm@gmail.com for any questions about code (i.e. usage, bugs, etc.)
# Function:		get.REst
# Purpose:		Returns Rank-based ICA estimator for Asymmetric ICA (Hallin and Mehta, manuscript available at arXiv:1304.3073 [stat.ME]) OR the Signed-rank ICA estimator for Symmetric ICA (Ilmonen and Paindaveine, AOS, 2011)
# Input Arg:	type = 'R' for R-estimator for mixing matrix using data-driven skew-t scores.
# 				type = 'SR' for signed-rank estimator for mixing matrix using data-driven Student t scores.
#				Xn = kXn matrix of k-variate mixed data with n observations in columns for model:  Xn = L %*% Zn with kXk population mixing matrix L and kXn matrix Zn of iid random variables whose rows represent source signals
#				PE = Preliminary estimator
#				max.shape = maximum absolute value of shape parameter for skew-t scores.
# 				min.df = smallest degrees of freedom for skew-t scores (in R-estimator) and Student t scores in (Signed-Rank estimator)
#				if.parallel = True or False. If Parallel = TRUE, R package multicore is used parallelize the computation of the cross-information quanities
# Output:		Est = List of three matricesmin.df
#				Est$PE = Original Preliminary 
#				Est$Ln.PE = Value of Original Preliminary mapped to Parameter space of observationally equilalent mixing matrices
#				Est$REst = R-estimator
# Dependencies:	R Packages: multicore, sn, MASS

require(multicore); require(sn); require(MASS)

get.REst <- function(type,Xn,PE,max.shape=15,min.df=3, if.parallel=T){
	n = ncol(Xn); m = nrow(Xn)
	Ln.PE = standardizeM(PE)
	Zn.1 = ginv(Ln.PE) %*% Xn
	Zn = Zn.1 - matrix(apply(Zn.1,1,median),m,n)
	if(type=='R'){
		score = scores.st(Zn,50,max.shape=max.shape,min.df=min.df)   
	}else if (type=='SR'){
		score = fit.t(Zn,min.df=min.df)
	}

	# Getting Maximal Invariant
	# If type=='R', returns marginal ranks divided by n + 1
	# If type=='SR', returns (1/2 + SR/(2*(n+1)))*I[sign=+1]  + (1/2 - SR/(2*(n+1)))*I[sign=-1] where SR = marginal signed-ranks (marginal ranks of residuals' absolute value)
	R = get.MI(Zn,type)
	# Getting Rank Scores
	T0 = rank.score(R,type,score)
	Nhat = est.CIQ(Xn,Ln.PE,T0,type,score,if.parallel=if.parallel)

	REst = Ln.PE + n^(-1/2) * Ln.PE %*% (Nhat - diag(diag(Ln.PE %*% Nhat)))
	REst = standardizeM(REst)
	Est = list(PE=PE,Ln.PE=Ln.PE,REst=REst)
	return(Est)
}


# Functions

get.MI <- function(Zn,type='R'){
	n = ncol(Zn); m = nrow(Zn)
	if(type=='R'){
		R = apply(Zn,1,rank)/(n+1)
	}else if(type=='SR'){
		sR = apply(abs(Zn),1,rank,ties='first')/(2*(n+1)) + matrix(1/2,n,m)
		S = apply(Zn + matrix(runif(n*m,-1,1)/10^6,m,n),1,sign) # Multiply by 100 to ensure no signs = 0
		R = sR*(S==1) + (1 - sR)*(S==-1)
	}
	return(t(R))
}

rank.score <- function(R,type='R',score){
	m = nrow(R); n = ncol(R); 		
	T0 = matrix(0,m,m);
	Jf = matrix(0,m,n); Fi = matrix(0,m,n)
	mu.Jf = matrix(0,m,1); mu.Fi = mu.Jf
	if(type=='R'){
		z0 = score[[1]]; cdf = score[[2]]; phi = score[[3]]
		for(j in 1:m){
			zi = approx(x=cdf[j,], y =z0[j,], xout=R[j,])$y
			Jf[j,] = approx(x= z0[j,], y = phi[j,], xout=zi)$y
			Fi[j,] = zi
		}
		mu.Jf = matrix(apply(Jf,1,mean),m,1)
		mu.Fi = matrix(apply(Fi,1,mean),m,1)
	}else{
		for(j in 1:m){
			nuhat = score[j]
			q = qt(R[j,],nuhat)
			if(nuhat == Inf){
				Jf[j,] = q
			}else{
				Jf[j,] = (nuhat + 1)*q/(nuhat + q*q)
			}
			Fi[j,] = q	
		}
	}

	for(i in 1:n){
		T0 = T0 + matrix(Jf[,i],m,1) %*% matrix(Fi[,i],1,m)/n
	}
	T0 = sqrt(n)*(T0 - mu.Jf %*% t(mu.Fi))
	T0 = T0 - diag(diag(T0))
	return(T0)
}


scores.st <- function(Zn,nT,max.shape=15,min.df=3){
	m = nrow(Zn); n = ncol(Zn)
	z0 = array(0,c(m, 2*nT+1)); phi = z0; cdf = z0;
	f0=array(0,m);
	for(j in 1:m){
		Zj = Zn[j,];
		fp = st.mle(y=Zj)$dp
		if(fp[4]<=min.df){
			fp = st.mle(y=Zj,fixed.df=min.df)$dp
		}
		if(max.shape < Inf){
			fp[3] = sign(fp[3])*min(max.shape,abs(fp[3]))
		}
		fp[2] = 1
		fp[1] = -qst(0.5,dp=fp)
		scores = scores.st1(fp,nT)			
		z0[j,] = scores[[1]]
		cdf[j,] = scores[[2]]
		phi[j,] = scores[[3]]
		f0[j] = scores[[4]]
				
	}
	return(list(z0,cdf,phi,f0))
}

# ----- ----- ----- ----- ----- ----- 
scores.st1 <- function(fp, nT){
	require(sn)
	lim = round(qst(c(4e-4,1-4e-4),dp=fp),4);
	iL = list((1:nT),((nT+2):(2*nT+1)))
	z0 = array(0,2*nT+1);
	phi = z0;
	for(k in 1:2){
		eps = abs(lim[k])/nT;
		if(k==1){
			s0 = -eps * rev(iL[[1]])
		}else{
			s0 = eps * iL[[1]]
		}
		z0[iL[[k]]] = s0
		phi[iL[[k]]] = phi.st1(fp,s0,eps)
	}
	phi[nT+1] = phi.st1(fp,0,mean(abs(lim))/nT)
	cdf = pst(z0,dp=fp)
	return(list(z0,cdf,phi,dst(0,dp=fp)))
}


# ----- ----- ----- ----- ----- ----- 
phi.st1 <- function(fp,s0,eps){
	require(sn);
	d0 = dst(s0,dp=fp)
	eps.v = 0.5*rep(eps,length(s0))
	dp = dst(s0 + eps.v,dp=fp  )
	dn = dst(s0 - eps.v,dp=fp  )
	dpr = (dp - dn)/eps
	
	phi = array(0,length(s0))
	for(i in 1:length(s0)){
		if(d0[i] > 1e-15 ){
			phi[i] = -dpr[i]/d0[i]
		}else{
			phi[i] = 0
		}
	}
	return(phi)
}

# Fit Skew t
fit.skewtDist <-function(x, min.df=3, ...) 
{
    start = c(mu=0, sigma = 1, alpha=0.1, nu = 4)
    
    loglik.skewt = function(x, theta, min.df) {
        f = -sum(dst(x,location=theta[1],scale=theta[2],shape=theta[3],df=max(3,theta[4]),log=T))
        return(f)
    }
    fit = nlminb(start = start, objective = loglik.skewt, lower = c(-Inf,0,-Inf,min.df), upper = c(Inf, Inf,Inf,Inf), x = x, min.df = min.df, control=list(eval.max = 30, iter.max=25))
    
    fit.skt = fit$par
    fit.skt[1] = 0;
    fit.skt[2] = 1;
    fit.skt[3] = sign(fit.skt[3])*min(15,abs(fit.skt[3]))
    fit.skt[1] = -qst(0.5,dp=fit.skt)

    return(fit.skt)
}

# Fit Df t
fit.t <- function(Zn.L1,min.df=3){
	m = nrow(Zn.L1)
	fit.t = array(0,m)
	for(j in 1:m){
		fp = fit.df.tDist(Zn.L1[j,],min.df)	
		fit.t[j] = fp[4]	
	}
	return(fit.t)
}


fit.df.tDist <-function(x, min.df=3, ...) 
{
	x = x
    start = c(sd = sqrt(var(x)), nu = 4)
    
    loglik = function(x, theta, min.df) {
        f = -sum(log(dstd1(x, theta[1], theta[2], min.df)))
        return(f)
    }
    fit = nlminb(start = start, objective = loglik, lower = c(
        0, min.df), upper = c(Inf, Inf), x = x, min.df = min.df, ..., control=list(eval.max = 20, iter.max=15))
    
   	sd.hat = max(0.1,fit$par[1])
    nu.hat = max(min.df,fit$par[2])
    if(nu.hat > 1e03){
      nu.hat=Inf; scale = sd.hat
    }else{
      scale = sd.hat*sqrt((nu.hat-2)/nu.hat)      
    }

    return(c(0, scale, 0, nu.hat))
}

dstd1 = function(x,sd=1,nu=5,min.df){
		nu = max(nu,min.df)
      	sd = max(sd,0.01)
    	s=min( sqrt(nu/(nu-2)), 10e04)
    	z = x/sd
    	result = dt(x=z*s, df=nu) * (s/sd)
    	return(result)
}



obj.CIQ <- function(lambda,trs,Xn,Ln.PE,T0,type,score){
	t = trs[1]; r = trs[2]; s= trs[3]
	m = nrow(Xn); n = ncol(Xn)
	er = array(0, m); er[r] = 1
	es = array(0, m); es[s] = 1
	L.int = n^(-1/2) * Ln.PE %*% (er %*% t(es) - diag(diag(Ln.PE %*% er %*% t(es))))
	
	if(t==1){ r1 = r; s1 = s}else{
		r1 = s; s1 = r;
	}
	L2.1 = Ln.PE + lambda * T0[r1,s1] * L.int
	L2 = standardizeM(L2.1)
	if(sum(abs(round(L2.1,6) - round(L2,6))) > 0.001){
		L2 = Ln.PE
	}
	
	Zn.2 = ginv(L2) %*% Xn 
	if(type=='R')
	Zn.2 = Zn.2 - matrix(apply(Zn.2,1,median),m,n) #+ matrix(runif(n*m,0,1)/10^5,m,n) # This randomization is added just to avoid ties when getting maximal invariant.
	R.2 = get.MI(Zn.2,type)
	rank.scores = rank.score(R.2,type,score)

	return(T0[r1,s1] * rank.scores[r1,s1])

}


opt.CIQ <- function(trs,Xn,Ln.PE,T0,type,score){
	 m = nrow(Xn); n = ncol(Xn);
	bp = c(seq(0,1,0.1),seq(2,10,1))
	inc = c(rep(0.1,10),rep(1,10))	
	ep = bp + inc	
	flag = F

	for(cnt in 1:length(bp)){
		fb = obj.CIQ(bp[cnt],trs,Xn,Ln.PE,T0,type,score)
		fe = obj.CIQ(ep[cnt],trs,Xn,Ln.PE,T0,type,score)
		#print(paste(cnt,bp[cnt],fb,ep[cnt],fe))
			if(sign(fb) != sign(fe)){
				flag= T;
				break;
			}
	}
	if(all(flag=F,type == 'R')){
		#print('flag')
		bp = -rev(ep)
		ep = bp + rev(inc)
		for(cnt in 1:length(bp)){
			fb = obj.CIQ(bp[cnt],trs,Xn,Ln.PE,T0,type,score)
			fe = obj.CIQ(ep[cnt],trs,Xn,Ln.PE,T0,type,score)
			if(sign(fb) != sign(fe)){
				flag= T;
				break;
			}
		}
	}
	
	if(flag){
		y1 = fb; y2 = fe
		x1 = bp[cnt]; x2 = ep[cnt];
		slope = (y1 - y2)/(x1 - x2)
		inter = y1 - slope*x1
		ciqi = -inter/slope
		Hrc = sign(ciqi)*min(abs(1/ciqi),100)	
	}else{
		Hrc = 0.1
	}	
	return(Hrc)
}

est.CIQ <- function(Xn,Ln.PE,T0,type,score,if.parallel=T){
	m = nrow(Xn)
	trsL = c(NULL)
	for(r in 1:m){for(s in 1:m){
			if(r != s){
				trsL = c(trsL,list(c(1,r,s))); #,c(2,r,s)))
				trsL = c(trsL,list(c(2,r,s))); #,c(2,r,s)))
			}
	}}
	if(if.parallel){
		CIQ = mclapply(X=trsL,opt.CIQ,Xn=Xn,Ln.PE=Ln.PE,T0=T0,type=type,score=score)
	}else if(!if.parallel){
		CIQ = lapply(X=trsL,opt.CIQ,Xn=Xn,Ln.PE=Ln.PE,T0=T0,type=type,score=score)		
	}
	CNT = length(trsL)
	gamma = matrix(0,m,m)
	rho = matrix(0,m,m)
	for(c1 in 1:CNT){
		trs = trsL[[c1]]
		if(trs[1] == 1){
			gamma[trs[2],trs[3]] = CIQ[[c1]]
		}else{
			rho[trs[2],trs[3]] = CIQ[[c1]]			
		}
	}
	Alpha = matrix(0,m,m)
	Beta = matrix(0,m,m)
	Denom = matrix(0,m,m)
	for(r in 1:m){for (s in 1:m){if(r != s){
				Denom[r,s] = gamma[r,s]*gamma[s,r] - rho[r,s]*rho[s,r]
				if(Denom[r,s] != 0){
					Alpha[r,s] = gamma[r,s]/Denom[r,s]
					Beta[r,s] = -rho[r,s]/Denom[r,s]
				}
	}}}
	
	Nhat = t(Alpha) * T0 + t(Beta)*t(T0)
	return(Nhat)
}



standardizeM <- function(M){
	m = ncol(M);
	M2 = M
	for(j in 1:m){
		M2[,j] = M2[,j]/(sqrt(sum(M2[,j]*M2[,j])))
	}
	aM2 = abs(M2); M3 = matrix(0,m,m)
	R = matrix(0,m,m); 
	R[1,] = 1; E = R
	P = matrix(0,m);
	for(c in 1:m){ 
		rs = 1;
		for(r in 2:m){
			if(max(aM2[1:r,c]) == aM2[r,c] ){
				R[r,c] = 2*r - rs;
				rs = rs + r;
				E[r, c] = r;
			}		
		}
	}
	R2 = colSums(R); R2 = rank(R2, ties.method='first')
	for(k in 1:m){
		M3[,R2[k]] = M2[,k];
	}
	D2 = 1/diag(M3);
	M4 = M3 %*% diag(D2)
	return(M4)
}


