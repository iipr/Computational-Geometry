# -*- coding: utf-8 -*-
 
from __future__ import division
import numpy as np
 
class Fisher:
    '''Implements the Fisher discrimiation
    method to classify points'''
 
    def __init__(self):
        '''The basic constructor'''
        self.w = None
        self.c = None
 
    def compute_umbral(self, X0, X1, vmu0, vmu1):
        '''Computes the 'c' for the 
        Fisher's method discriminant'''
    
        #Computes the standar desviation of proyected points
        sigma0 = np.std(np.dot(self.w, X0.transpose()))
        sigma1 = np.std(np.dot(self.w, X1.transpose()))
 
        #Calculate the proyected means
        mu0 = self.w.dot(vmu0.transpose())
        mu1 = self.w.dot(vmu1.transpose())
 
        #Computes shape of points
        N0 = X0.shape[0]
        N1 = X1.shape[0]
        N = N0+N1
 
        #Computes probabily of a class
        p0 =N0/N
        p1 = N1/N
        #Computes the coeficients for the Fisher's umbral 
        c = (mu1*sigma0**2 - mu0*sigma1**2 - np.sqrt(2*sigma1**2*(np.log(p0/sigma0)-np.log(p1/sigma1)) - 2*sigma0**2*np.log(p0/sigma0) + 2*sigma0**2*np.log(p1/sigma1) + mu0**2 - 2*mu0*mu1 + mu1**2)*sigma0*sigma1)/(sigma0**2 - sigma1**2)
        return c
 
    def compute_Sw(self, X0, X1):
        #Computes shape of points
        N0 = X0.shape[0]
        N1 = X1.shape[0]

         
        S0 = np.cov(X0, y=None, rowvar=0, ddof=N0-1)
        S1 = np.cov(X1, y=None, rowvar=0, ddof=N1-1)
        return (S0+S1)
 
    def train_fisher(self, X0, X1):
        #Convertimos a tipo array que nos gusta mas. Si ya es array no hace nada.
        X0 = np.asarray(X0)
        X1 = np.asarray(X1)
        #Compute the vectorial means
        mu0 = np.mean(X0, axis=0)
        mu1 = np.mean(X1, axis=0)
 
        #Compute the Sw matrix
        Sw = self.compute_Sw(X0, X1)
         
        #Calculalmos el vector w (sin normalizar)
        w = np.linalg.solve(Sw,mu1-mu0)
        wnorm = np.linalg.norm(w)
        # El resultado es el vector w normalizado
        self.w = w/wnorm
 
        self.c = self.compute_umbral(X0, X1, mu0, mu1).astype(float)
 
    def classify_fisher(self, X):
        '''Clasify de points X using the w matrix
        and the umbral c calculated by the Fisher
        method, acording to the training points'''
        X = np.asarray(X)
        #Proyectamos los puntos.
        y = np.dot(self.w, X.transpose())
 
        #Devolvemos una lista de enteros. A la clase 0 si 
        # son menores que c y a la clase 1 otherwise.
        clasificacion = (y > self.c).astype(int)
 
        return clasificacion.tolist()
