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

        #Computes Sw (Like at fisher.py)

        #Computes Sb (la nueva)

        #Computes the eighenvalues and eighenvectors (eigh.eigh(Sb, Sw))

        #Sort the eighenvectors (np.argsort)

        #Changes to inverse order (a[::-1])

        #Computes the W matrix (taking the eighvectors corresponding 
        #to the highest 'reduced_dim' eighenvalues)
        
        #OJO_1: Kike es un petacas.
        #OJO_2: Proyectar con con la base ortonormal del subespacio generado por los autovectores. 

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
