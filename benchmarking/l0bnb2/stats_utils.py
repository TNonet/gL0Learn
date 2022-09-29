import numpy as np



def compute_likelihood_given_suff(Y,Theta,R):
    n,p = Y.shape
    cost = 0
    for i in range(p):
        cost += np.log(Theta[i,i])
        cost -= R[:,i]@R[:,i]/Theta[i,i]
    return cost

def compute_likelihood(X, Theta, assume_centered = False):
    n,p = X.shape
    if not assume_centered:
        X = X - np.mean(X, axis=0)
    Y = X/np.sqrt(n)
    R = Y@Theta
    return compute_likelihood_given_suff(Y,Theta,R)