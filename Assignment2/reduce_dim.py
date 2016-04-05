# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np 
from scipy.linalg import eigh 

#For testing purposes
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import lda
from sklearn.decomposition import PCA as sklpca


class LDA:
    def __init__(self):
        self.w = None
        self.mean = None

    def fit(self, X_train, y_train, reduced_dim):

        N = np.shape(X_train)[0]  
        D = np.shape(X_train)[1]  
        p = np.size(X, axis=1)  # columns of training dataset
        
        Sw = np.zeros((D,D))  
        #Sb = np.zeros((D,D)) #Nos podemos ahorrar declararla creo
        St = np.cov(X_train.transpose(), bias = 1) * N
          
        #Label the classes in order to loop on them
        labels = np.unique(y_train)
        #Count the number of points on each class
        Nk = np.bincount(y_train)
        #print labels  
        #print Nk
        #print range(len(labels))
        for Ck in range(len(labels)):   
            Xk = X_train[y_train == Ck, :]
            Sw = np.add(Sw, np.cov(Xk.transpose(), bias = 1) * Nk[Ck])
            
        Sb = np.subtract(St, Sw)  
        
        diagonal = np.diag(Sw)
        meanDiag = diagonal.mean(axis=0)
        delta = 0.001*meanDiag
        deltaI = delta*np.eye(p)
        evals, evecs = eigh(Sw)
        indices = np.argsort(evals)  
        if np.abs(evals[0]) < 1e-7:
            Sw = Sw + deltaI        
        
        
        evals, evecs = eigh(Sb, Sw) 
        #evals, evecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb)) 
        indices = np.argsort(evals)[::-1]
       
        #indices = indices  
        evecs = evecs[:,indices]  
        evals = evals[indices]  
        evecs = evecs[:,:reduced_dim]  
        evecs /= np.apply_along_axis(np.linalg.norm, 0, evecs)

        self.mean = np.mean(X_train, axis = 0)
        self.w = evecs
        #OJO: Proyectar con la base ortonormal del subespacio generado por los autovectores. 

    def transform(self, X):
        """Project data to maximize class separation.
        Parameters:
        ----------
        X
            Array of points to be projected
        Returns:	
        ----------
        X_new
            Array projected from X input array using the W"""

        # Centre data  
        X = -X + self.mean
        #Project the points of X
        X_new = X.dot(self.w)
        return X_new

class PCA:
    def __init__(self):
        self.w = None

    def _covariance_matrix(self, X):
        mean = np.mean(X, axis = 0)
        St = (X - mean).T.dot((X - mean)) / (X.shape[0] - 1)
        return St

    def _orderedeigenvecs(self, St):
        evals, evecs = np.linalg.eig(St)
        sorted_ind = np.argsort(evals)[::-1]
        evecs = evecs[sorted_ind]
        return evecs

    def fit(self, X_train, reduced_dim):
        St = self._covariance_matrix(X_train)
        evecs = self._orderedeigenvecs(St)
        self.w = np.vstack([evecs[:, i] for i in range(reduced_dim)]).T
        return self

    def transform(self, X):
        return X.dot(self.w)

if __name__ == "__main__" :
    #Cargamos datos de nuestro dataset
    db = datasets.load_digits()
    X, y = db.data, db.target

    #Cargamos lda para 2 componentes 
#    fisher_LDA = lda.LDA(n_components=2)
    fisher_PCA = sklpca(n_components=2)

    #Hacemos el entrenamiento con lda
#    fisher_LDA.fit(X, y)
    fisher_PCA.fit(X)

    #Hacemos la reduccion lda con los mismos puntos
 #   X_reduced = fisher_LDA.transform(X)
    X_reduced = fisher_PCA.transform(X)

    #Pintamos todos nuestros puntos
    plt.plot(X_reduced[:, 0], X_reduced[:, 1], 'o')

    #Pintamos los 1s, los 8s y los 0s de distinto color
    for k in [0, 1, 8]:
        plt.plot(X_reduced[y == k, 0], X_reduced[y == k, 1], 'o')
    plt.axis([-40,40,-30,30])
    plt.figure()
    #plt.show()

    #LDA tests, examples from datasets.
#    mi_lda = LDA()
#    mi_lda.fit(X, y, 2)
#    X_reduced_1 = mi_lda.transform(X)
#    plt.plot(X_reduced_1[:, 0], X_reduced_1[:, 1], 'o')
#    for k in [0, 1, 8]:
#        plt.plot(X_reduced_1[y == k, 0], X_reduced_1[y == k, 1], 'o')
#    plt.show()

    #PCA tests, examples from datasets.
    mi_pca = PCA()
    mi_pca.fit(X, 2)
    X_reduced_2 = mi_pca.transform(X)
    plt.plot(X_reduced_2[:, 0], X_reduced_2[:, 1], 'o')
    for k in [0, 1, 8]:
        plt.plot(X_reduced_2[y == k, 0], X_reduced_2[y == k, 1], 'o')
    #plt.figure()
    plt.show()
