#this file contains the functions used in the Power Iteration PCA analysis

import numpy as np
from scipy.spatial.distance import cdist

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


def IntraClusterDistance(centroid, pc1,pc2): #returns the average distance between centroid its cluster points
    total_distance = 0
    count = len(pc1)
    for i in range(len(pc1)):
        distance = cdist([centroid], [[pc1[i],pc2[i]]])[0][0] #Euclidan distance
        total_distance += distance
    return total_distance/count

def GetInterclusterDistance(centroids):
    interclusterdistances = []
    for i in range(len(centroids)):
        distance = 0
        for j in range(len(centroids)):
            if i != j:
                distance += cdist([centroids[i]], [centroids[j]])[0][0]
        interclusterdistances.append(distance/(len(centroids)-1)) #average distance between centroids
    return interclusterdistances

def GetCentroids(top10PCs, k, num_samples):
    centroids = []
    for i in range(k):
        avgpc1 = top10PCs[0][i*num_samples:(i+1)*num_samples].mean()
        avgpc2 = top10PCs[1][i*num_samples:(i+1)*num_samples].mean()
        centroids.append([avgpc1, avgpc2])
    return centroids


def GetIntraClusterDistance(centroids, top10PCs, k, num_samples):

    intraclusterdistances = []

    for i in range(k):
        pc1 = top10PCs[0][i*num_samples:(i+1)*num_samples]
        pc2 = top10PCs[1][i*num_samples:(i+1)*num_samples]
        intraclusterdistances.append(IntraClusterDistance(centroids[i], pc1, pc2))

    return intraclusterdistances

#Get the ratio of intercluster distance to intracluster distance. K is the number of clusters
def GetDistanceRatio(top10PCs, k, num_samples):
    centroids = GetCentroids(top10PCs, k, num_samples)
    intraclusterdistances = GetIntraClusterDistance(centroids,top10PCs, k, num_samples)
    interclusterdistances = GetInterclusterDistance(centroids)
    return np.array(interclusterdistances)/np.array(intraclusterdistances)