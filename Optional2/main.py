import numpy as np
import matplotlib.pyplot as plt

import fisher as fh
import loadPoints as lp
import LSQ_classification as lsq

class Main:
    
    def __init__(self):
    	self.trainer = lp.LoadPoints()
        self.points, self.classes = self.trainer.get_points()
        self.points = np.asarray(self.points)
        self.classes = np.asarray(self.classes)
        self.labels = np.unique(self.classes)
        self.numClasses = np.shape(self.labels)[0]

        self.computeLSQ()
        self.compute_fisher()

        self.loadFigureResult()
    	
        self.drawPoints()
        self.drawContour()

        plt.show()

    def loadFigureResult(self):
        self.fig = plt.figure(facecolor='g',figsize=(100,50))
        #self.fig.set_label('Classify')
        self.plot_lsq = self.fig.add_subplot(121, title='Least squares')
        self.plot_fisher = self.fig.add_subplot(122, title='Fisher')
        self.configureAxis(-20, 20, -20, 20)

    def configureAxis(self, a, b, c, d):
    	"""Configura los limites de los ejes 
    	que mostraremos de forma [a, b]x[c,d]"""
	#En un futuro la clase almacenara tb estos limites
    	self.plot_lsq.axis([a, b, c, d])
        self.plot_fisher.axis([a, b, c, d])


    def compute_fisher(self):
        #First we adapt points to Fisher input      
   
        self.X_0 = self.points[self.classes < self.numClasses/2, :]
        self.X_1 = self.points[self.classes >= self.numClasses/2, :]

        #Then we train
        self.fisher = fh.Fisher()
        self.fisher.train_fisher(self.X_0, self.X_1)


    def computeLSQ(self):
        self.lsq_classifier = lsq.LSQ_classification()
        self.lsq_classifier.compute_W(self.points.transpose(), self.classes, self.numClasses)

    def drawPoints(self):
        for Ck in self.labels:
            self.plot_lsq.plot(self.points[self.classes == Ck,0], self.points[self.classes == Ck,1], 'o')

        self.plot_fisher.plot(self.X_0[:,0], self.X_0[:,1], 'bo')
        self.plot_fisher.plot(self.X_1[:,0], self.X_1[:,1], 'go')

    def drawContour(self):
    	'''Utiliza la W para mostrar la frontera entre clases'''
    	#Creamos una malla de puntos 1000x1000 que clasificaremos
    	# con nuestra W y luego pintaremos las fronteras.
    	Xs = np.linspace(-20, 20, 1000)
    	Ys = np.linspace(-20, 20, 1000)
    	XX, YY = np.meshgrid(Xs, Ys)

    	#Aplanamos los puntos para usarlos en classify
    	x, y = XX.flatten(), YY.flatten()
    	pts = np.vstack((x, y))

    	#Contour requiere una matriz, no un array plano,
    	#por eso usamos reshape. 

        ZZ_lsq = self.lsq_classifier.classifier(pts).reshape(1000, 1000)

    	ZZ_fisher = np.asarray(self.fisher.classify_fisher(pts.transpose())).reshape(1000, 1000)

    	plt.ion()

        self.plot_lsq.contour(Xs, Ys, ZZ_lsq)

    	self.plot_fisher.contour(Xs, Ys, ZZ_fisher)

        plt.ioff()

if __name__ == '__main__':
    m = Main()

