# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np 
from scipy.linalg import eigh 


class LDA:
    def __init__(self):
        self.w = None
        self.mean = None

    def fit(self, X_train, y_train, reduced_dim):

        N = np.shape(X_train)[0]
        D = np.shape(X_train)[1]
        
        Sw = np.zeros((D,D))
        St = np.cov(X_train.T, bias = 1) * N
          
        #Label the classes in order to loop on them
        labels = np.unique(y_train)
        #Count the number of points on each class
        Nk = np.bincount(y_train)
        for Ck in labels:   
            #Xk = X_train[y_train == Ck, :] #<-- Esta linea se podria eliminar poniendo su valor en la siguiente
            Sw = np.add(Sw, np.cov(X_train[y_train == Ck, :].T, bias = 1) * Nk[Ck])
            
        Sb = np.subtract(St, Sw)  
        
        evals, evecs = eigh(Sb, Sw)
        indices = np.argsort(evals)[::-1]
       
        #indices = indices  
        evecs = evecs[:,indices]  
        #evals = evals[indices]  
        evecs = evecs[:,:reduced_dim]  
        #evecs /= np.apply_along_axis(np.linalg.norm, 0, evecs)

        self.mean = np.mean(X_train, axis = 0)
        self.w = evecs

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
        X = X - self.mean
        #Project the points of X
        X_new = X.dot(self.w)
        return X_new

class PCA:
    def __init__(self):
        self.w = None
        self.mean = None

    def _covariance_matrix(self, X):
        #mean = np.mean(X, axis = 0)
        St = np.cov(X.T, bias = 1) * np.shape(X)[0]
        #St = (X - mean).T.dot((X - mean))# / (X.shape[0] - 1)
        return St

    def _orderedeigenvecs(self, St):
        #evals, evecs = np.linalg.eig(St)
        evals, evecs = eigh(St)
        sorted_ind = np.argsort(evals)[::-1]
        evecs = evecs[:,sorted_ind]
        return evecs

    def fit(self, X_train, reduced_dim):
        St = self._covariance_matrix(X_train)
        evecs = self._orderedeigenvecs(St)
        self.w = evecs[:,:reduced_dim]  
        #self.w = np.vstack([evecs[:, i] for i in range(reduced_dim)]).T
        self.mean = np.mean(X_train, axis = 0)
        return self

    def transform(self, X):
        # Centre data  
        X = X - self.mean
        return X.dot(self.w)