import numpy as np
import matplotlib.pyplot as plt

import fisher as fh
import loadPoints as lp
import LSQ_classification as lsq

class Main:
    """Creates a window in which you can enter points
    belonging to different classes. Once you're done, 
    pressing 'enter' causes a classification using least
    squares and another using Fisher.
    Fisher splits in two classes, so the points of the first half of
    the classes go to the first class and the points of the second half 
    of the classes go to the other class."""


    def __init__(self):
        #Loads an array of points and an array of classes from the user
        self.trainer = lp.LoadPoints()

        #Obtains the datas introduced by the usser
        self.points, self.classes = self.trainer.get_points()
        self.points = np.asarray(self.points)
        self.classes = np.asarray(self.classes)

        # Computes an array with diferent labels of the classes
        self.labels = np.unique(self.classes)

        # Indicates the number of classes
        self.numClasses = np.shape(self.labels)[0]

        # Computes the matrix of LSQ method with the points introduced by the user
        self.computeLSQ()
        # Computes the matrix of Fisher method with the points introduced by the user
        self.compute_fisher()

        # Loads the second plot figure, where we compare the methods
        self.loadFigureResult()

        # Plots the points introduced by the user
        self.drawPoints()
        
        # Plots the contours of the classes obtained with the training
        self.drawContour()

        plt.show()


    def loadFigureResult(self):
        """Creates the figure where we compare the methods
        and configure it"""

        self.fig = plt.figure(facecolor='g', figsize=(100, 50))
        # self.fig.set_label('Classify')
        self.plot_lsq = self.fig.add_subplot(121, title='Least squares')
        self.plot_fisher = self.fig.add_subplot(122, title='Fisher')
        self.configureAxis(-20, 20, -20, 20)

    def configureAxis(self, a, b, c, d):
        """Configure the axis of the subplots [a, b] x [c,d]"""
        self.plot_lsq.axis([a, b, c, d])
        self.plot_fisher.axis([a, b, c, d])


    def compute_fisher(self):
        """Fits the w matrix of Fisher method with the points
        introduced mannualy by the user."""
        # First we adapt points to Fisher input

        # The points of the first half of classes goes to the X_0 class
        self.X_0 = self.points[self.classes < self.numClasses/2,:]
        # The points of the second half of classes goes to the X_1 class
        self.X_1 = self.points[self.classes >= self.numClasses/2,:]

        # Then we train the w matrix of the Fisher method with the users points
        self.fisher = fh.Fisher()
        self.fisher.train_fisher(self.X_0, self.X_1)


    def computeLSQ(self):
        """Fits the w matrix of Least Squares method with the points
        introduced mannualy by the user."""

        self.lsq_classifier = lsq.LSQ_classification()
        self.lsq_classifier.compute_W(self.points.transpose(), self.classes, self.numClasses)

    def drawPoints(self):
        """Plots the user points in both subplots"""
        #Plots the points of the LSQ method in its subplot
        for Ck in self.labels:
            self.plot_lsq.plot(self.points[self.classes == Ck, 0], self.points[self.classes == Ck, 1], 'o')

        #Plots the points of the Fisher method in its subplot
        self.plot_fisher.plot(self.X_0[:, 0], self.X_0[:, 1], 'bo')
        self.plot_fisher.plot(self.X_1[:, 0], self.X_1[:, 1], 'go')

    def drawContour(self):
        '''Plots the borders of the classes, according to the classification
        obtained after fit the method with user data. Using the w matrix'''

        # We create a grid with 1.000.000 of points.
        Xs = np.linspace(-20, 20, 1000)
        Ys = np.linspace(-20, 20, 1000)
        XX, YY = np.meshgrid(Xs, Ys)

        # We adapt the points in order to be used correctly
        x, y = XX.flatten(), YY.flatten()
        pts = np.vstack((x, y))

        # Contour requires a matrix, not an flatten array,
        # that is why we use reshape.

        # We classify the 1.000.000 points using lsq method, and 
        # store their class in the ZZ_lsq variable.
        ZZ_lsq = self.lsq_classifier.classifier(pts).reshape(1000, 1000)

        # We classify the 1.000.000 points using Fisher method, and 
        # store their class in the ZZ_fisher variable.
        ZZ_fisher = np.asarray(self.fisher.classify_fisher(pts.transpose())).reshape(1000, 1000)

        # We plots the border of the classes of the 1.000.000 points classified using lsq
        self.plot_lsq.contour(Xs, Ys, ZZ_lsq)

        # We plots the border of the classes of the 1.000.000 points classified using fisher
        self.plot_fisher.contour(Xs, Ys, ZZ_fisher)

if __name__ == '__main__':
    m = Main()
