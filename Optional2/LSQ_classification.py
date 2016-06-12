import numpy as np
from numpy import linalg as la

class LSQ_classification:
    
    def __init__(self):
    	#Inicializamos la W y el subplot para pintar.
   	#Hay cosas que podriamos tambien inicializar.
        self.W = None
 
    def compute_W(self, X, tags, K):
        '''Computes the matrix of coefficients of the K 
        affine forms used for the linear square clasification
        of points as explained in Bishop, ch.4.1
        X: list of training points [[x0, y0], ...]
        tags: list of classes to which the data points belong
              [k0, k1, ...]
        K: number of different classes'''
        #Convertimos a tipo array que nos gusta mas. Si ya es array no hace nada.
        X = np.asarray(X)

    	#Calculamos el numero de puntos que hemos recibido
        N = X.shape[1]
     	#Como en Bishop, la X_tilde anyade unos
        X_tilde = np.vstack((np.ones(N), X))
        T = np.zeros((N, K))
    	#Ponemos un 1 en la clase k a la que pertenece el punto,
    	#el resto seran ceros. Si pertenece a la clase 2: (0010..0)
    	#Esto se hace para cada punto, generando la matriz T (ver Bishop)
        for row, tag in enumerate(tags):
            T[row, tag] = 1
    	
    	#La funcion resuelve el sistema de minimos cuadrados(puede usarse para comprobar):
        #self.W = np.linalg.lstsq(X_tilde.transpose(), T)[0]
    	
    	#Calculamos la pseudoinversa de X_tilde
    	pseudo = np.linalg.inv(X_tilde.dot(X_tilde.T))

    	#Obtenemos la matriz W de clasificacion por minimos cuadrados
    	self.W = np.dot(pseudo.dot(X_tilde), T)
    
    def classifier(self, points):
        '''Clasifica los puntos dados en las clases,
    	para ello utiliza la W obtenida con
    	 los puntos de entrenamiento'''
        if self.W == None:
            print 'First train the classifier!'
            return
        points = np.asarray(points)
        M = points.shape[1]
    	#Anyadimos unos para completar la X_tilde a clasificar
        pts_tilde = np.vstack((np.ones(M), points))

    	#De acuerdo con el Bishop, obtenemos la matriz de 
    	#clasificacion de los puntos. Nos quedamos con el 
    	#valor k mas alto de y_k para cada punto
        Y = (self.W.transpose()).dot(pts_tilde)
        return np.argmax(Y, axis=0)

    
    

    
    
    
