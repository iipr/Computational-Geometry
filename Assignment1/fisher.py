# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np


class Fisher:

    '''Implements Fisher's lineal discriminant
    method to classify points (see Bishop p.186)'''

    def __init__(self):
        '''Basic constructor'''
        self.w = None
        self.c = None

    def compute_umbral(self, X0, X1, vmu0, vmu1):
        '''Computes the value c for
        Fisher's lineal discriminant'''

        # Computes the standard deviation of the projected points
        sigma0 = np.std(np.dot(self.w, X0.transpose()))
        sigma1 = np.std(np.dot(self.w, X1.transpose()))

        # Computes the projected means
        mu0 = self.w.dot(vmu0.transpose())
        mu1 = self.w.dot(vmu1.transpose())

        # Computes the number of points on each class
        N0 = X0.shape[0]
        N1 = X1.shape[0]
        N = N0 + N1

        # Computes the probabily of each class
        p0 = N0 / N
        p1 = N1 / N
        # Computes the coefficients for the threshold
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
        # Computes the number of points on each class
        N0 = X0.shape[0]
        N1 = X1.shape[0]

        S0 = np.cov(X0, y=None, rowvar=0, ddof=N0 - 1)
        S1 = np.cov(X1, y=None, rowvar=0, ddof=N1 - 1)
        return (S0 + S1)

    def train_fisher(self, X0, X1):
        # Turns lists into arrays for latter purposes
        # Nothing happens if they were arrays already
        X0 = np.asarray(X0)
        X1 = np.asarray(X1)
        # Compute vectorial means
        mu0 = np.mean(X0, axis=0)
        mu1 = np.mean(X1, axis=0)

        # Computes matrix S_w
        Sw = self.compute_Sw(X0, X1)

        # Computes vector w
        w = np.linalg.solve(Sw, mu1 - mu0)
        wnorm = np.linalg.norm(w)
        # And normalize it
        self.w = w / wnorm

        self.c = self.compute_umbral(X0, X1, mu0, mu1).astype(float)

    def classify_fisher(self, X):
        '''Classify the points given by X using matrix w
        and the threshold c calculated by Fisher's
        method, according to the training points'''
        X = np.asarray(X)
        # Computes projected points
        y = np.dot(self.w, X.transpose())

        # Returns a list of 0-1 values
        # Class 0: y <= c
        # Class 1: y > c
        clasificacion = (y > self.c).astype(int)

        return clasificacion.tolist()
