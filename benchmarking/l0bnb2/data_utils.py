import numpy as np
import scipy.stats as st
from scipy.sparse import diags



"""
Theta = B + delta*I
delta chosen to have condition(Theta)=cond
Each coordinate of B is set to zero or 0.5 with probability 1-p0 and p0, respectively.
"""

def generate_rothman(n,p, p0, cond, normalize="precision",rng=None):
    
    assert normalize in {"covariance", "precision"}
    if rng is None:
        rng = np.random
    elif type(rng) == int:
        rng = np.random.RandomState(rng)
    B = 0.5*rng.binomial(1,p0,(p,p))
    B = B*np.tri(p,p,-1)
    np.fill_diagonal(B, 0)
    B = (B + np.transpose(B))
    evals = np.linalg.eigvalsh(B)
    delta = (cond*evals[0]-evals[p-1])/(1-cond)
    Theta = B + delta*np.identity(p)
    Sigma = np.linalg.inv(Theta)
    if normalize == "covariance":
        q = Sigma[1,1]
        Sigma = Sigma/q
        Theta = Theta*q
    else:
        q = Theta[1,1]
        Sigma = Sigma*q
        Theta = Theta/q
        
    X = rng.multivariate_normal(np.zeros(p), Sigma, n)
    
    return X, Sigma, Theta
   



def generate_independent(n,p,normalize="precision",rng=None):
    assert normalize in {"covariance", "precision"}
    if rng is None:
        rng = np.random
    elif type(rng) == int:
        rng = np.random.RandomState(rng)
    
    X = rng.randn(n,p)
    Sigma = Theta = np.eye(p)
    if normalize == "covariance":
        return X, Sigma, Theta
    else:
        diag = np.diag(Theta)
        X *= np.sqrt(diag)
        Sigma = Sigma*np.sqrt(diag)*np.sqrt(diag[:,None])
        Theta = Theta/np.sqrt(diag)/np.sqrt(diag[:,None])
    return X, Sigma, Theta


def generate_constant_correlation(n,p,rho=0,normalize="precision",rng=None):
    assert normalize in {"covariance", "precision"}
    if rng is None:
        rng = np.random
    elif type(rng) == int:
        rng = np.random.RandomState(rng)
    
    X = rng.randn(n, p) 
    if rho != 0:
        X = X*np.sqrt(1-rho) + np.sqrt(rho)*rng.randn(n,1)
    Sigma = rho*np.ones((p,p))
    np.fill_diagonal(Sigma,1)
    
    pcorr = -rho/(1+(p-2)*rho)
    precision = 1/(1+(p-1)*rho*pcorr)
    
    Theta = pcorr*np.ones((p,p))
    np.fill_diagonal(Theta,1)
    Theta = Theta*precision
    if normalize == "covariance":
        return X, Sigma, Theta
    else:
        diag = np.diag(Theta)
        X *= np.sqrt(diag)
        Sigma = Sigma*np.sqrt(diag)*np.sqrt(diag[:,None])
        Theta = Theta/np.sqrt(diag)/np.sqrt(diag[:,None])
    return X, Sigma, Theta





def generate_expoential_precision(n,p,rho=0.6,normalize="precision",rng=None):
    assert normalize in {"covariance", "precision"}
    if rng is None:
        rng = np.random
    elif type(rng) == int:
        rng = np.random.RandomState(rng)
        

    Theta = np.power(rho,np.abs(np.arange(p)-np.arange(p)[:,None]))
    Sigma = np.linalg.inv(Theta) 
    X = rng.multivariate_normal(np.zeros(p), Sigma, n)
    
    if normalize == "precision":
        return X, Sigma, Theta
    else:
        return X, Sigma, Theta




def generate_Toeplitz_correlation(n,p,rho=0,normalize="precision",rng=None):
    assert normalize in {"covariance", "precision"}
    if rng is None:
        rng = np.random
    elif type(rng) == int:
        rng = np.random.RandomState(rng)
        
    X = rng.randn(n, p)
    q = np.sqrt(1-rho**2)
    if rho != 0:
        for i in range(1,p):
            X[:,i] = X[:,i-1]*rho+q*X[:,i]
    Sigma = np.power(rho,np.abs(np.arange(p)-np.arange(p)[:,None]))
    
    Theta = np.where(np.abs(np.arange(p)-np.arange(p)[:,None])==1, -rho/(1-rho**2),0)
    np.fill_diagonal(Theta, (1+rho**2)/(1-rho**2))
    Theta[0,0] = Theta[-1,-1] = 1/(1-rho**2)
    if normalize == "covariance":
        return X, Sigma, Theta
    else:
        diag = np.diag(Theta)
        X *= np.sqrt(diag)
        Sigma = Sigma*np.sqrt(diag)*np.sqrt(diag[:,None])
        Theta = Theta/np.sqrt(diag)/np.sqrt(diag[:,None])
    return X, Sigma, Theta


def generate_banded_partial_correlation(n,p,rho=0,normalize="precision",rng=None):
    assert normalize in {"covariance", "precision"}
    assert rho >= -0.5
    if rng is None:
        rng = np.random
    elif type(rng) == int:
        rng = np.random.RandomState(rng)
        
    X = rng.randn(n, p)
    k = [np.full(p-1,rho),np.ones(p),np.full(p-1,rho)]
    offset = [-1,0,1]
    Theta = diags(k,offset).toarray()
    
    Sigma = np.linalg.inv(Theta)
    
    C = np.linalg.cholesky(Sigma)
    X = X@C.T
    if normalize == "precision":
        return X, Sigma, Theta
    else:
        diag = np.diag(Sigma)
        X /= np.sqrt(diag)
        Sigma = Sigma/np.sqrt(diag)/np.sqrt(diag[:,None])
        Theta = Theta*np.sqrt(diag)*np.sqrt(diag[:,None])
    return X, Sigma,Theta


def generate_regression(n,p,k,val=1,normalize="precision",rng=None):
    assert normalize in {"covariance","precision"}
    assert k <= p-1
    if rng is None:
        rng = np.random
    elif type(rng) == int:
        rng = np.random.RandomState(rng)
        
    X = np.random.randn(n,p-1)
    beta = np.zeros(p-1)
    beta[np.random.choice(p-1,k,replace=False)] = (2*np.random.choice(2,k)-1)*val
    noise = np.random.randn(n)
    y = X@beta+noise
    X = np.hstack([y[:,None],X])
    
    Sigma = np.eye(p)
    Sigma[0,0] = 1 + np.sum(beta**2)
    Sigma[0,1:]=beta
    Sigma[1:,0]=beta
    
    Theta = np.eye(p)
    Theta[1:,1:] += beta[:,None]@beta[None,:]
    Theta[0,1:]=-beta
    Theta[1:,0]=-beta
    
    if normalize == "precision":
        diag = np.diag(Theta)
        X *= np.sqrt(diag)
        Sigma = Sigma*np.sqrt(diag)*np.sqrt(diag[:,None])
        Theta = Theta/np.sqrt(diag)/np.sqrt(diag[:,None])
    else:
        diag = np.diag(Sigma)
        X /= np.sqrt(diag)
        Sigma = Sigma/np.sqrt(diag)/np.sqrt(diag[:,None])
        Theta = Theta*np.sqrt(diag)*np.sqrt(diag[:,None])
    return X, Sigma,Theta
    
    
def generate_synthetic(n,p,model="independent",normalize="precision", rng=None, **kwargs):
    """
    Generate synthetic data set
    Parameters
    ----------
    model: str
        "independent", "constant_correlation", "Toeplitz_correlation", "AR1", "banded_partial_correlation", or "regression"
        "independent": independent samples
        "constant_correlation": samples with constant correlation "rho" given by kwargs
        "Toeplitz_correlation" or "AR1": samples with correlation rho^{|i-j|}, where 'rho' is given by kwargs
        "banded_partial_correlation": samples with banded partial correlation rho if |i-j|=1, where 'rho' is given by kwargs
        "regression": the first covariate is a linear model of k of the others, the coefficient is randomly selected from \pm val, where 'k' and 'val' are given by kwargs  
    normalize: str, default "precision"
        "covariance" or "precision". How to normalize the data so that either covariance or precision matrix has diagonal 1
    rng: None, int, or random generator
        If rng is None, then it becomes np.random
        If rng is int, then it becomes np.random.RandomState(rng)
    Returns
    -------
    X:  n x p numpy array
        simulated data
    Sigma: p x p numpy array
        population covariance matrix of sampled data
    Theta: p x p numpy array
        population precision matrix of sampled data
    """
    
    assert model in {"independent", "constant_correlation", "Toeplitz_correlation", "AR1", "banded_partial_correlation", "regression", "rothman", "exponential"}
    if model == "independent":
        return generate_independent(n,p,normalize,rng)
    elif model == "constant_correlation":
        rho = kwargs.get("rho", 0.5)
        return generate_constant_correlation(n,p,rho,normalize,rng)
    elif model in {"Toeplitz_correlation","AR1"}:
        rho = kwargs.get("rho", 0.5)
        return generate_Toeplitz_correlation(n,p,rho,normalize,rng)
    elif model == "banded_partial_correlation":
        rho = kwargs.get("rho", -0.5)
        return generate_banded_partial_correlation(n,p,rho,normalize,rng)
    elif model == "rothman":
        p0 = kwargs.get("p0", 0.2)
        cond = kwargs.get("cond", 2)
        return generate_rothman(n,p, p0, cond, normalize,rng )
    elif model == "exponential":
        rho = kwargs.get("rho", 0.6)
        return generate_expoential_precision(n,p,rho,normalize,rng)
    elif model == "regression":
        k = kwargs.get("k", np.round(np.sqrt(p)))
        val = kwargs.get("val", 1)
        return generate_regression(n,p,k,val,normalize,rng)
    return 



def preprocess(X, assume_centered=False, cholesky=False):
    if assume_centered:
        X_mean = np.zeros(X.shape[1])
    else:
        X_mean = np.mean(X, axis=0)
        X = X - X_mean


    n, p = X.shape
    if cholesky and n > p:
        S = X.T@X/n
        S_diag = np.diag(S)
        Y = np.linalg.cholesky(S).T
    else:
        Y = X/np.sqrt(n)
        S_diag = np.linalg.norm(Y, axis=0) ** 2
    return n,p,X,X_mean,Y,S_diag


"""
Centering and normalizing the training, validation and test data 
"""


def preprocess2(X, X_val, X_test, assume_centered=False):
    if assume_centered:
        X_mean = np.zeros(X.shape[1])
    else:
        X_mean = np.mean(X, axis=0)
        Y = X - X_mean
        Y_val = X_val - X_mean
        Y_test = X_test - X_mean

    n, p = Y.shape
    Y = Y/np.sqrt(n)
    n, p = Y_val.shape
    Y_val = Y_val/np.sqrt(n)
    n, p = Y_test.shape
    Y_test = Y_test/np.sqrt(n)
    

    return X,X_mean,Y,Y_val, Y_test