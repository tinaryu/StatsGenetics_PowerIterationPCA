#this file contains the functions used in the Power Iteration PCA analysis

import numpy as np


def calc_af(geno): 
    '''Calculate the allele frequencies for each row in the geno'''
    return geno.mean(axis=1).filled(-1) / 2

def GetCovarianceMatrixPsi(geno):
    # calculate allele frequency and standard deviation
    p = calc_af(geno)
    SD = np.sqrt(2*p*(1-p)) # assuming Hardy-Weinberg equilibrium

    # normalize genotypes
    X = ((geno - 2*p[:,None]) / SD[:,None]).filled(0)

    # remove monomorphic rows
    X = X[SD != 0,]
    # calculate the covariance matrix
    psi = X.transpose().dot(X) / float(X.shape[0])
    return psi

def normalize_vec(v): # *
    '''normalize vector to have unit length'''
    v_to_return = v/np.sqrt((v**2).sum())
    return v_to_return

# multiply with covariance matrix and renormalize until convergence
def RunUntilConverge(b,psi):
    num_iters = 0 
    while(True):
        vec_old = b
        b = normalize_vec(psi.dot(vec_old))

        current_diff = ((b - vec_old)**2).mean()

        if current_diff < 1e-16:
            print("Power iteration: converged at iter="+str(num_iters))
            break
        if num_iters > 100000:
            print("Power iteration: did not converge")
            break
        num_iters+=1
    return b

def GetTop10PCs(psi):
    '''Get the top 10 principal components'''
    b = normalize_vec(np.random.rand(len(psi)))
    pcs = []
    for i in range(10):
        b = RunUntilConverge(b, psi)
        pcs.append(b)
        d = b.T @ psi @ b
        psi = psi - d * np.outer(b, b)
    return pcs

def RunPowerIteration10PCs(geno):
    psi = GetCovarianceMatrixPsi(geno)
    return GetTop10PCs(psi)

