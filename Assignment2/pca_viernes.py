# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np 
from scipy.linalg import eigh 

#for tests
import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn import lda


class LDA:
    def __init__(self):
        self.w = None
        self.r_dim = None

    def fit(self, X_train, y_train, reduced_dim):

           # Centre data  
        data -= data.mean(axis=0)  
        nData = shape(data)[0]  
        nDim = shape(data)[1]  
          
        Sw = zeros((nDim,nDim))  
        Sb = zeros((nDim,nDim))  
          
        St = cov(transpose(data))  
          
        # Loop over classes  
        classes = unique(labels)  
        for i in range(len(classes)):  
            # Find relevant datapoints  
            indices = squeeze(where(labels==classes[i]))  
            d = squeeze(data[indices,:])  
            Sw += np.cov(d.transpose(),bias=1)*shape(indices)[0]
              
        Sb = St - Sw  
        # Now solve for W  
        # Compute eigenvalues, eigenvectors and sort into order  
        #evals,evecs = linalg.eig(dot(linalg.pinv(Sw),sqrt(Sb)))  
        evals,evecs = eigh(Sw,Sb)  
        indices = argsort(evals)  
        indices = indices[::-1]  
        evecs = evecs[:,indices]  
        evals = evals[indices]  
        w = evecs[:,:redDim]  
        #print evals, w  
        self.w = w

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

        #Project the points of X
        X_new = self.w.dot(X.transpose())

        return X_new

class PCA:
    
    def fit(self, X_train, reduced_dim):
        pass #incluye aquí tu código

    def transform(self, X):
        pass #incluye aquí tu código

if __name__ == "__main__" :


    """Lda tests, examples from datasets.
    In our asignment: 
    Fm = LDA()
    Fm.fit(X, y, 2)
    Fm.transform(X)"""

    db = datasets.load_digits()

    #Cargamos lda para 2 componentes 
    fisher_LDA = lda.LDA(n_components=2)

    #Cargamos datos de nuestro dataset
    X, y = db.data, db.target

    #Hacemos el entrenamiento con lda
    fisher_LDA.fit(X, y)

    #Hacemos la reduccion lda con los mismos puntos
    X_reduced = fisher_LDA.transform(X)

    #Pintamos todos nuestros puntos
    plt.plot(X_reduced[:, 0], X_reduced[:, 1], 'o')

    #Pintamos los 1s, los 8s y los 0s de distinto color
    for k in [0, 1, 8]:
        plt.plot(X_reduced[y == k, 0], X_reduced[y == k, 1], 'o')

    plt.show()