# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np 
from scipy.linalg import eigh 


class LDA:
    """Linear discriminant analysis for dimensionality reduction.

    First, the matrix Sw , St and Sb  are computed using the set of 
    input training points given by X_train and the y_train vector.

    Using the Sw and Sb matrix we compute the projecting matrix w 
    making large the between-class covariance and small the within-class covariance.

    Afterwards, the points given by X are projected using the w matrix
    in order to reduce the dimensionality.

    Note
    ----
    LDA is also called "Fisher’s discriminant for multiple classes".
    More information about the method can be found at Bishop (page 191 and after).

    Attributes
    ----------
    w : ndarray[float]
        Projection matrix.
    mean : float
        Mean of X_train points.

    Methods
    -------
    fit
        Computes the projection matrix w.
    transform
        Projects the data, reducing dimensionality using the w computed in fit.

    """
    def __init__(self):
        self.w = None
        self.mean = None

    def fit(self, X_train, y_train, reduced_dim):
        """Computes the projection matrix w for lda.

        Parameters
        ----------
        X_train
            Set of training points.
        y_train
            Vector of classes of X_train points.
        reduced_dim
            Target dimension of the reduction.

        Examples
        --------

        """
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
        """Project the points of X using the 
        w matrix calculated with fit

        Parameters
        ----------
        X
            Set of points to be projected.

        Returns:	
        ----------
        X_new
            Array projected from X points using the w
        
        Examples
        --------

        """

        # Centre data before projecting
        X = X - self.mean
        #Project the points of X
        X_new = X.dot(self.w)
        return X_new

class PCA:
    """Principal component analysis for dimensionality reduction.

    First, computes the covariance matrix of the set of 
    input training points given by X_train. Then, obtains the projection
    matrix according to the pca method.

    Afterwards, the points given by X are projected using the w matrix
    in order to reduce the dimensionality.

    Note
    ----
    More information about the method can be found at Bishop (page 561 and after).

    Attributes
    ----------
    w : ndarray[float]
        Projection matrix.
    mean : float
        Mean of X_train points.

    Methods
    -------
    _covariance_matrix
        Returns the covariance matrix St.
    _orderedeigenvecs
        Returns the ordered eigenvectors of a given matrix. 
    fit
        Computes the projection matrix w of the pca method.
    transform
        Projects the data, reducing dimensionality using the w computed in fit.

    """
    def __init__(self):
        self.w = None
        self.mean = None

    def _covariance_matrix(self, X):
        """Computes the coovariance 
        matrix of the given points.

        Parameters
        ----------
        X
            Set of points.
        
        Returns:	
        ----------
        St
            Covariance matrix of the X points.
  
        Examples
        --------

        """
        #mean = np.mean(X, axis = 0)
        St = np.cov(X.T, bias = 1) * np.shape(X)[0]
        #St = (X - mean).T.dot((X - mean))# / (X.shape[0] - 1)
        return St

    def _orderedeigenvecs(self, St):
        """Computes the eigenvectors of 
        the given matrix and order it.

        Parameters
        ----------
        St
            Matrix.
        
        Returns:	
        ----------
        evecs
            Matrix of ordered eigenvectors.
  
        Examples
        --------

        """
        #evals, evecs = np.linalg.eig(St)
        evals, evecs = eigh(St)
        sorted_ind = np.argsort(evals)[::-1]
        evecs = evecs[:,sorted_ind]
        return evecs

    def fit(self, X_train, reduced_dim):
        """Computes the projection matrix w for pca.

        Parameters
        ----------
        X_train
            Set of training points.
        reduced_dim
            Target dimension of the reduction.

        Examples
        --------

        """
        St = self._covariance_matrix(X_train)
        evecs = self._orderedeigenvecs(St)
        self.w = evecs[:,:reduced_dim]  
        #self.w = np.vstack([evecs[:, i] for i in range(reduced_dim)]).T
        self.mean = np.mean(X_train, axis = 0)
        return self

    def transform(self, X):
        """Project the points of X using the 
        w matrix calculated with fit

        Parameters
        ----------
        X
            Set of points to be projected.

        Returns:	
        ----------
        X_new
            Array projected from X points using the w
        
        Examples
        --------

        """
        # Centre data
        X = X - self.mean
        # Project data points
        X_new = X.dot(self.w)
        return X_new
