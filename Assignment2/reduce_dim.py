# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np 
from scipy.linalg import eigh 

#for tests
import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn import lda


class LDA:
    
    def fit(self, X_train, y_train, reduced_dim):
        pass #incluye aquí tu código

    def transform(self, X):
        pass #incluye aquí tu código

class PCA:
    
    def fit(self, X_train, reduced_dim):
        pass #incluye aquí tu código

    def transform(self, X):
        pass #incluye aquí tu código

if __name__ == "__main__" :

    FisherMulticlase = LDA()
    "Pruebas de lda"
    db = datasets.load_digits()

    db.keys()

    db.data[0].shape

    db.images[0]

    db.data[0]

    plt.imshow(db.images[0], cmap='gray')

    fisher_LDA = lda.LDA(n_components=2)

    X, y = db.data, db.target

    fisher_LDA.fit(X, y)

    X_reduced = fisher_LDA.transform(X)

    X_reduced.shape

    plt.plot(X_reduced[:, 0], X_reduced[:, 1], 'o')

    y == 7

    for k in [1, 8, 0]:
        plt.plot(X_reduced[y == k, 0], X_reduced[y == k, 1], 'o')

    plt.show()
