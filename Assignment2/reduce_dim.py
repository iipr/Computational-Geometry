# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np 
from scipy.linalg import eigh 

#For testing purposes
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import lda
#from scipy.linalg import eig 


class LDA:
    def __init__(self):
        self.w = None
        self.mean = None
        #self.r_dim = None

    def fit(self, X_train, y_train, reduced_dim):

        #Computes Sw (Like at fisher.py)

        #Computes Sb (la nueva)

        #Computes the eighenvalues and eighenvectors (eigh.eigh(Sb, Sw))

        #Sort the eighenvectors (np.argsort)

        #Changes to inverse order (a[::-1])

        #Computes the W matrix (taking the eighvectors corresponding 
        #to the highest 'reduced_dim' eighenvalues)

        # Centre data  
        #data -= data.mean(axis=0)  

        nData = np.shape(X_train)[0]  
        nDim = np.shape(X_train)[1]  
          
        Sw = np.zeros((nDim,nDim))  
        #Sb = np.zeros((nDim,nDim))  
          
        St = np.cov(X_train.transpose(), bias = 1) * nData
          
        #Label the classes and loop on them
        labels = np.unique(y_train)
        #print labels
        #Count the number of points on each class
        Nk = np.bincount(y_train)  
        #print Nk
        #print range(len(labels))
        for Ck in range(len(labels)):  
            # Find relevant datapoints  
            #indices = np.squeeze(np.where(y_train == classes[i]))  
            #print indices
            #d = np.squeeze(X_train[indices,:])   
            #otra forma para ver los puntos de la cada clase
            Xk = X_train[y_train == Ck, :]
            #print Xk
            Sw = np.add(Sw, np.cov(Xk.transpose(), bias = 1) * Nk[Ck])
            #print np.cov(Xk.transpose(), bias = 1)
            #print np.shape(np.cov(Xk.transpose(), bias = 1))[0], np.shape(np.cov(d.transpose(), bias = 1))[1]
              
        Sb = np.subtract(St, Sw)  
        #print np.shape(Sw)[0], np.shape(Sw)[1]
        #print Sw
        # Now solve for W  
        # Compute eigenvalues, eigenvectors and sort into order  
        #evals,evecs = linalg.eig(dot(linalg.pinv(Sw),sqrt(Sb)))  
        evals, evecs = eigh(Sb, Sw) 
        #evals, evecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb)) 
        indices = np.argsort(evals)  
        indices = indices[::-1]  
        evecs = evecs[:,indices]  
        evals = evals[indices]  
        evecs = evecs[:,:reduced_dim]  
        #print evals, w  
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
        X = X - self.mean
        #Project the points of X
        X_new = X.dot(self.w)

        return X_new

class PCA:
    
    def fit(self, X_train, reduced_dim):
        pass #incluye aquí tu código

    def transform(self, X):
        pass #incluye aquí tu código

if __name__ == "__main__" :

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

    #plt.show()

    #Lda tests, examples from datasets.
    
    mi_lda = LDA()
    mi_lda.fit(X, y, 2)
    X_reduced_1 = mi_lda.transform(X)
    plt.plot(X_reduced_1[:, 0], X_reduced_1[:, 1], 'o')
    for k in [0, 1, 8]:
        plt.plot(X_reduced_1[y == k, 0], X_reduced_1[y == k, 1], 'o')

    plt.show()
