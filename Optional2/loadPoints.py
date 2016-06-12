import numpy as np
import matplotlib.pyplot as plt

class Punto:
    """Contiene las coordenadas de cada punto y la clase a la que pertenece"""
    def __init__(self, xc, yc, cl):
        self.x = xc
        self.y = yc
        self.point_class = cl
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def getClass(self):
        return self.point_class

class Color:
    """Controls colors of points"""

    def __init__(self):
        self.colors = ['ro', 'go', 'bo', 'co', 'mo']
        self.actual = 0
        self.clases = 0 #Indica el numero de colores en uso
        self.used = set()

    def changeColor(self):
        if self.actual > 3:
            self.actual = 0
        else:
            self.actual = self.actual+1

    def getColor(self):
        return self.colors[self.actual]

    def addColor(self, color):
	if color not in self.used:
            self.used.add(color)
            self.clases=self.clases+1

    def getUsed(self):
        return self.used

    def getClass(self):
        return self.actual

class LoadPoints:
    """We introduce bidimensional points of several classes in order to train classify algorithms"""

    def __init__(self):
        self.fig = plt.figure()
        self.cuadro = self.fig.add_subplot(111)
        self.cuadro.axis([-20, 20, -20, 20])

        self.points = []
        self.classes = []

        self.colors = Color()

        self.cidButton = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
	self.cidKey = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

	plt.show()
	

    def onclick(self, event):
        actual_color = self.colors.getColor()
        if event.button == 1:

            self.cuadro.plot(event.xdata, event.ydata, actual_color)
            self.cuadro.figure.canvas.draw()
 
            self.points.append([event.xdata, event.ydata])
            self.classes.append(self.colors.getClass())

            if actual_color not in self.colors.getUsed():
                self.colors.addColor(actual_color)

        elif event.button == 3:
            self.colors.changeColor()

    def on_key(self, event):
        if event.key == 'enter':
            plt.close()

    def get_points(self):
        """Returns an arrray with the points and another with the class to which each belongs"""
        return self.points, self.classes       

