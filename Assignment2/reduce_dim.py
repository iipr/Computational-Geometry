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
        Vectorial mean of X_train points.

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
        """Compute w, the projection matrix for LDA.

        Parameters
        ----------
        X_train
            Set of training points.
        y_train
            Vector of classes of the points given by X_train.
        reduced_dim
            Target dimension of the dimensionality reduction.

        Examples
        --------
            >>> my_lda = LDA()
        Introduce some values for X_train and y_train:
            >>> X_train = np.asarray([[34.12, 31.05, -5.98], [-17.21, -28.20, 17.54],
                                      [36.76, 26.94, -6.18], [-37.79, -1.14, -1.370],
                                      [34.65, 29.18, -6.29], [-18.57, 28.660, 18.45]])
            >>> y_train = np.asarray([0, 1, 2, 2, 1, 0])
        Choose the target dimension of reduction:
            >>> reduced_dim = 2
            >>> my_lda.fit(X_train, y_train, reduced_dim)
        Then w will be:
            >>> [[-0.00325751 -0.02306827]
                 [ 0.0325941   0.0152601 ]
                 [ 0.04380306 -0.05359897]]

        """
        # Compute and save the vectorial mean for latter purposes
        self.mean = np.mean(X_train, axis=0)
        # Count the total number of training points and their dimension
        N, D = np.shape(X_train)

        Sw = np.zeros((D, D))
        # Compute the covariance matrix of X_train
        St = np.cov(X_train.T, bias=1) * N

        # Label the classes in order to loop on them
        labels = np.unique(y_train)
        # Count the number of points on each class
        Nk = np.bincount(y_train)
        for Ck in labels:
            Sw = np.add(Sw, np.cov(X_train[y_train == Ck, :].T, bias=1) * Nk[Ck])

        # Compute between-class covariance matrix
        Sb = np.subtract(St, Sw)
        # Compute eigenvalues for LDA problem and sort them in a decreasing fashion
        self.w = np.fliplr(eigh(Sb, Sw, eigvals = (D - reduced_dim, D - 1))[1])


    def transform(self, X):
        """Project the points of X using w, which was calculated with fit.

        Parameters
        ----------
        X
            Set of points to be projected.

        Returns:
        ----------
        X_projected
            Projected points using w.

        Examples
        --------
        Assuming w is already computed, introduce some values for X:
            >>> X = np.asarray([[-36.61, -0.95, -27.53],
                                [34.06, 25.760, -5.190], [-16.56, -28.34, 19.55],
                                [-16.71, -27.70, 18.27], [-36.91, -0.46, -26.70],
                                [-36.65, -3.62, -23.23], [-18.72, 28.57, 19.04]])
            >>> X_projected = my_lda.transform(X)
        X_projected:
            >>> [[-1.68814683  2.35296383]
                 [-0.06920616 -0.06707466]
                 [-0.58396431 -1.05096876]
                 [-0.61868337 -0.96913537]
                 [-1.63484192  2.32287462]
                 [-1.58668962  2.08266652]
                 [ 1.25566287 -0.10535326]]

        """
        # Centre and project data points
        return (X - self.mean).dot(self.w)


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
        """Compute w, the projection matrix for PCA.

        Parameters
        ----------
        X_train
            Set of training points.
        reduced_dim
            Target dimension of the dimensionality reduction.

        Examples
        --------
            >>> my_pca = PCA()
        Introduce some values for X_train:
            >>> X_train = np.asarray([[34.12, 31.05, -5.98], [-17.21, -28.20, 17.54],
                                      [36.76, 26.94, -6.18], [-37.79, -1.14, -1.370],
                                      [34.65, 29.18, -6.29], [-18.57, 28.66, 18.450]])
        Choose the target dimension of reduction:
            >>> reduced_dim = 2
            >>> my_pca.fit(X_train, reduced_dim)
        The projecting matrix w:
            >>> [[ 0.83558706  0.50419046]
                 [ 0.50454667 -0.86141214]
                 [-0.21731757 -0.06132779]]
        With some other values for X_train:
            >>> X_train = np.asarray([[34.12, 31.05, -5.98,-17.21, -28.20],
                                      [36.76, 26.940, -6.18, -1.14, -1.37],
                                      [34.65, 29.180, -6.29, 28.66, 18.45]])
            >>> my_pca.fit(X_train, reduced_dim)
        The new w will be:
            >>> [[ 0.00642278  0.28546226]
                 [-0.02630362 -0.38638449]
                 [-0.00469732 -0.00663974]
                 [ 0.70451509 -0.63060545]
                 [ 0.70915674  0.60951702]]

        """
        # Compute the dimension of the training points
        D = np.shape(X_train)[1]
        # Compute and save the vectorial mean for latter purposes
        self.mean = np.mean(X_train, axis=0)

        # Compute the covariance matrix of X_train
        St = np.cov(X_train.T, bias=1) * np.shape(X_train)[0]
        # Compute eigenvalues for PCA problem and sort them in a decreasing fashion
        self.w = np.fliplr(eigh(St, eigvals = (D - reduced_dim, D - 1))[1])


    def transform(self, X):
        """Project the points of X using w, which was calculated with fit.

        Parameters
        ----------
        X
            Set of points to be projected.

        Returns:
        ----------
        X_projected
            Projected points using w.

        Examples
        --------
        Assuming w is already computed, introduce some values for X:
            >>> X = np.asarray([[-36.61, -0.95, -27.53], [34.060, 25.760, -5.19],
                                [-16.56, -28.34, 19.55], [-16.71, -27.70, 18.27],
                                [-36.91, -0.46, -26.70], [-36.65, -3.62, -23.23],
                                [-18.72, 28.570, 19.04]])
            >>> X_projected = my_pca.transform(X)
        X_projected will be:
            >>> [[-36.22567184  -6.05483728]
                 [ 31.44683248   5.19792137]
                 [-43.52299595  24.76094785]
                 [-43.04725765  24.21251507]
                 [-36.40949367  -6.67908843]
                 [-38.54070049  -4.03874396]
                 [-16.50328106 -25.31979152]]
        Now with a different X:
            >>> X = np.asarray([[-36.61, -0.950, -27.53, 25.76, -5.19],
                                [-16.56, -28.34, 19.55, -27.70, 18.27]])
            >>> X_projected = my_pca.transform(X)
        X_projected will be:
            >>> [[ 15.10385223 -23.73764876]
                 [ -5.29462463  40.26777864]]

        """
        # Centre and project data points
        return (X - self.mean).dot(self.w)
