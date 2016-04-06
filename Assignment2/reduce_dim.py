# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from scipy.linalg import eigh


class LDA:
    """Linear Discriminant Analysis for dimensionality reduction.

    First, matrices Sw, St and Sb are computed using the set of
    input training points given by X_train and the class vector y_train.

    By using Sw and Sb, the projecting matrix w is computed leading to
    a big between-class covariance and a small within-class covariance.

    Afterwards, the points given by X are projected using w, providing
    a dimensionality reduction for an easier manipulation.

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

        Sw = np.zeros((D, D))
        St = np.cov(X_train.T, bias=1) * N

        # Label the classes in order to loop on them
        labels = np.unique(y_train)
        # Count the number of points on each class
        Nk = np.bincount(y_train)
        for Ck in labels:
            # Xk = X_train[y_train == Ck, :] #<-- Esta linea se podria eliminars poniendo su valor en la siguiente
            Sw = np.add(Sw, np.cov(X_train[y_train == Ck, :].T, bias=1) * Nk[Ck])

        Sb = np.subtract(St, Sw)
        evals, evecs = eigh(Sb, Sw)
        indices = np.argsort(evals)[::-1]

        evecs = evecs[:, indices]
        evecs = evecs[:, :reduced_dim]

        self.mean = np.mean(X_train, axis=0)
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
        # Project the points of X
        X_new = X.dot(self.w)
        return X_new


class PCA:
    """Principal Component Analysis for dimensionality reduction.

    First, compute St, the covariance matrix of the set of input
    training points given by X_train. Then, obtains the projection
    matrix according to the PCA method.

    Afterwards, the points given by X are projected using w, providing
    a dimensionality reduction for an easier manipulation.

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
    fit
        Computes the projection matrix w of the PCA method.
    transform
        Projects the data, reducing dimensionality using w (computed in fit).

    """

    def __init__(self):
        self.w = None
        self.mean = None

    def fit(self, X_train, reduced_dim):
        """Compute the projection matrix w for PCA.

        Parameters
        ----------
        X_train
            Set of training points.
        reduced_dim
            Target dimension of the dimensionality reduction.

        Examples
        --------

        """
        St = np.cov(X_train.T, bias=1) * np.shape(X_train)[0]
        evecs = eigh(St)[1]
        evecs = np.fliplr(evecs)
        self.w = evecs[:, :reduced_dim]
        self.mean = np.mean(X_train, axis=0)
        return self

    def transform(self, X):
        """Project the points of X using
        w matrix, calculated with fit.

        Parameters
        ----------
        X
            Set of points to be projected.

        Returns:
        ----------
        X_proj
            Projected points using the w.

        Examples
        --------

        """
        # Centre and project data points
        return (X - self.mean).dot(self.w)
