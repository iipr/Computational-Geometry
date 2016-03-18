# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np


class Fisher:
    """Fisher's lineal discriminant algorithm to classify points.

    First, the matrix Sw, the weight vector w and the threshold c are computed,
    depending on the sets of input training points given by X0 and X1.
    Afterwards, the points given by X are classified into one of the two
    classes (0 or 1) depending on the threshold c.

    Note
    ----
    More information about the method can be found at Bishop (page 186 and after).

    Attributes
    ----------
    w : ndarray[float]
        Weight vector.
    c : float
        Threshold for Fisher's method.

    Methods
    -------
    compute_threshold
        Return threshold c.
    compute_Sw
        Return matrix Sw.
    train_fisher
        Compute and update vector w and value c.
    classify_fisher
        Classify the points given by X according to the training points.

    """


    def __init__(self):
        """Instantiate w and c."""
        self.w = None
        self.c = None


    def compute_threshold(self, X0, X1, vmu0, vmu1):
        """Return threshold c for Fisher's lineal discriminant.

        Parameters
        ----------
        X0
            Set of training points for class 0.
        X1
            Set of training points for class 1.
        vmu0
            Vectorial mean for class 0.
        vmu1
            Vectorial mean for class 1.

        Returns
        -------
        c
            Threshold that gives the class distinction.

        Examples
        --------
        Use two sets of training points (X0 and X1):
            >>> Classifier = Fisher()
            >>> X0 = [[2, 3], [3, 5]]
            >>> X1 = [[4, 8], [5, 11]]
            >>> vmu0 = np.mean(X0, axis=0)
            >>> vmu1 = np.mean(X1, axis=0)
            >>> c = Classifier.compute_threshold(X0, X1, vmu0, vmu1)
        The threshold c will be:
            >>> 0.529218461347
        With different sets of training points:
            >>> X2 = [[1, 1], [2, 2]]
            >>> X3 = [[2, 1], [4, 2]]
            >>> vmu2 = np.mean(X2, axis=0)
            >>> vmu3 = np.mean(X3, axis=0)
            >>> c = Classifier.compute_threshold(X2, X3, vmu0, vmu1)
        The threshold c will be:
            >>> -0.0714285714286

        """
        # Compute the standard deviation of the projected points:
        sigma0 = np.std(np.dot(self.w, X0.transpose()))
        sigma1 = np.std(np.dot(self.w, X1.transpose()))

        # Compute the projected means:
        mu0 = self.w.dot(vmu0.transpose())
        mu1 = self.w.dot(vmu1.transpose())

        # Compute the number of points on each class and the total:
        N0 = X0.shape[0]
        N1 = X1.shape[0]
        N = N0 + N1

        # Compute the probability of each class:
        p0 = N0 / N
        p1 = N1 / N

        # Compute the coefficients for the threshold:
        c = (
            mu1 * sigma0**2 - mu0 * sigma1**2 - np.sqrt(
                2 * sigma1**2 * (
                    np.log(
                        p0 / sigma0) - np.log(
                            p1 / sigma1)) - 2 * sigma0**2 * np.log(
                                p0 / sigma0) + 2 * sigma0**2 * np.log(
                                    p1 / sigma1) + mu0**2 - 2 * mu0 * mu1 + mu1**2) * sigma0 * sigma1) / (
            sigma0**2 - sigma1**2)
        return c


    def compute_Sw(self, X0, X1):
        """Return matrix Sw for Fisher's lineal discriminant.

        Parameters
        ----------
        X0
            Set of training points for class 0.
        X1
            Set of training points for class 1.

        Returns
        -------
        Sw
            Total within-class covariance matrix.

        Examples
        --------
        Use two sets of training points (X0 and X1):
            >>> Classifier = Fisher()
            >>> X0 = [[2, 3], [3, 5]]
            >>> X1 = [[4, 8], [5, 11]]
            >>> Sw = Classifier.compute_Sw(X0, X1)
        The matrix Sw will be:
            >>> [[1.   2.5]
                 [2.5  6.5]]
        With different sets of training points:
            >>> X2 = [[1, 1], [2, 2]]
            >>> X3 = [[2, 1], [4, 2]]
            >>> Sw = Classifier.compute_Sw(X2, X3)
        The matrix Sw will be:
            >>> [[2.5  1.5]
                 [1.5  1. ]]

        """
        # Compute the number of points on each class:
        N0 = X0.shape[0]
        N1 = X1.shape[0]

        # Compute and return Sw = S0 + S1:
        S0 = np.cov(X0, y=None, rowvar=0, ddof=N0 - 1)
        S1 = np.cov(X1, y=None, rowvar=0, ddof=N1 - 1)
        return (S0 + S1)


    def train_fisher(self, X0, X1):
        """Compute and update vector w and value c for
        Fisher's lineal discriminant.

        Parameters
        ----------
        X0
            Set of training points for class 0.
        X1
            Set of training points for class 1.

        Examples
        --------
        Use two sets of training points (X0 and X1):
            >>> Classifier = Fisher()
            >>> X0 = [[2, 3], [3, 5]]
            >>> X1 = [[4, 8], [5, 11]]
            >>> Classifier.train_fisher(X0, X1)
        The values for w and c are, respectively:
            >>> [-0.83205029  0.5547002]
            >>> 0.529218461347
        With different sets of training points:
            >>> X2 = [[1, 1], [2, 2]]
            >>> X3 = [[2, 1], [4, 2]]
            >>> Classifier.train_fisher(X2, X3)
        The values for w and c are, respectively:
            >>> [0.5547002  -0.83205029]
            >>> -0.0714285714286

        """
        # Turn lists into arrays for latter purposes,
        # nothing happens if they were arrays already:
        X0 = np.asarray(X0)
        X1 = np.asarray(X1)

        # Compute vectorial means:
        mu0 = np.mean(X0, axis=0)
        mu1 = np.mean(X1, axis=0)

        # Compute matrix Sw with an auxiliar function:
        Sw = self.compute_Sw(X0, X1)

        # Compute vector w and normalize it:
        w = np.linalg.solve(Sw, mu1 - mu0)
        wnorm = np.linalg.norm(w)
        self.w = w / wnorm

        # Compute threshold c with an auxiliar function:
        self.c = self.compute_threshold(X0, X1, mu0, mu1).astype(float)


    def classify_fisher(self, X):
        """Classify the points given by X using matrix w
        and the threshold c calculated by Fisher's
        method, according to the training points.

        Parameters
        ----------
        X
            Set of points to classify according to Fisher's method.

        Returns
        -------
        clasification
            List of 0-1 values representing the class that each point belongs to.

        Examples
        --------
        First use some sets of training points (X0 and X1) and then classify X:
            >>> Classifier = Fisher()
            >>> X0 = [[2, 3], [3, 5]]
            >>> X1 = [[4, 8], [5, 11]]
            >>> Classifier.train_fisher(X0, X1)
            >>> X = [[1, 2], [3, 7], [5, 11], [2, 13]]
            >>> classes = Classifier.classify_fisher(X)
        The computed values (classes) will be:
            >>> [0, 1, 1, 1]
        With different sets of training points:
            >>> X2 = [[1, 1], [2, 2]]
            >>> X3 = [[2, 1], [4, 2]]
            >>> Classifier.train_fisher(X2, X3)
            >>> X = [[1, 2], [3, 2], [20, 30]]
            >>> classes = Classifier.classify_fisher(X)
        The computed values (classes) will be:
            >>> [0, 1, 0]

        """
        # Turn list into a array for latter purposes,
        # nothing happens if it was an array already:
        X = np.asarray(X)

        # Compute projected points:
        y = np.dot(self.w, X.transpose())

        # Returns a list of 0-1 values:
        # Class 0: y(k) <= c
        # Class 1: y(k) > c
        clasificacion = (y > self.c).astype(int)
        return clasificacion.tolist()
