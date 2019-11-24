'''
Dimensionality Reduction

@author: Abraham Zakharov
'''

import numpy as np
from sklearn.decomposition import PCA
from numpy import linalg as LA
import matplotlib.pyplot as pltcom
import matplotlib.pyplot as plt


def findpc(eigenvalues, percentage):
    #start r at 1
    r = 1
    current = 0
    while percentage > current:
        current = eigenvalues[0:r].sum() / sum(eigenvalues) * 100
        print('r: ', r)
        print ('Current % retained variance: ', current, '%')
        r = r + 1
    
def covar(data):

    dataTransposed = data.T
    dataTransposedTwo = dataTransposed
    covar = np.zeros((np.size(data,1), np.size(data,1)))
    for i, r in enumerate(dataTransposed):
        for j, c in enumerate(dataTransposedTwo):
            covar[i][j] = ((np.dot(r,c)) / (r.size))
            
    return covar

def center(data):
    means = (np.mean(data,0))
    dataTransposed = data.T
    
    #print(means)
    
    for i, r in enumerate(dataTransposed):
        for j, c in enumerate(r):
            dataTransposed[i,j] = c - means[i]
    
    return (dataTransposed.T)
    
if __name__ == '__main__':
    #part (a) import the data and center it
    data = np.genfromtxt('cloud.data')
    print('Original Dataset:')
    print(data)
    print('-----------------------------------------------------------------')
    print('Centered Dataset:')
    dataCentered = center(data)
    print(dataCentered)
    print('-----------------------------------------------------------------')
    #part (b) covariance
    print('Covariance of Dataset:')
    covarRes = covar(dataCentered)
    print(covarRes)
    print('-----------------------------------------------------------------')
    #part (c) eigenvectors/eigenvalues
    print('Eigenvectors and Eigenvalues of Dataset:')
    values, vectors = LA.eig(covarRes)
    print('Eigenvectors:')
    print(vectors)
    print('Eigenvalues:')
    print(values)
    print('-----------------------------------------------------------------')
    #part (d)
    print('Number of principal components (PCs) r that will ensure 90% retained variance:')
    findpc(values, 90)
    print('-----------------------------------------------------------------')
    #part (e) Plotting the first two components
    print('Plot the first two components')
    pltcom.title("PCA Cloud DB 2D Components")
    pltcom.xlabel("Dimensions")
    pltcom.ylabel("Magnitude")
    u1 = vectors[0]
    u2 = vectors[1]
    pltcom.xlim(1,10)
    pltcom.plot(np.append(np.roll(u1,1),u1[9]), label="u1")
    pltcom.plot(np.append(np.roll(u2,1),u2[9]), label="u2")
    plt.legend()
    pltcom.show()
    print('-----------------------------------------------------------------')    
    #part (f) Constructing the PCA
    print('PCA Computation with first 2 components')
    firstTwo = np.concatenate([[vectors[0]], [vectors[1]]])
    pca_2d = np.dot(dataCentered, firstTwo.T)
    print(pca_2d)
    plt.title("PCA Cloud DB 2D")
    plt.xlabel("ai1")
    plt.ylabel("ai2")
    plt.scatter(pca_2d[:,0], pca_2d[:,1])
    plt.show()
    print('-----------------------------------------------------------------')
    print('PCA using sklearn library:')
    #part (g) using sklearn library for PCA
    pca = PCA(n_components=10)
    pca.fit(data)  
    PCA(copy=True, iterated_power='auto', n_components=10, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
    print('Variance ratios:')
    print(pca.explained_variance_ratio_)  
    print('Eigenvalues:')
    print(pca.explained_variance_)  
    print('PCA Singular Values')
    print(pca.singular_values_) 
    pass
