import numpy as np
import matplotlib.pyplot as plt


class Color:
    """Auxiliar class to control colors of points.
       The color also idicates the classes of the points"""

    def __init__(self):
        """We have five different colors maximun"""
        self.colors = ['ro', 'go', 'bo', 'co', 'mo']
        # Indicates the color that we are using
        self.actual = 0
        # Indicates number of diferent classes that has been used
        self.clases = 0
        # Contains the colors that has been used
        self.used = set()

    def changeColor(self):
        """Changes the actual color to the next in the list"""
        if self.actual > 3:
            self.actual = 0
        else:
            self.actual = self.actual + 1

    def getColor(self):
        """Returns the actual color"""
        return self.colors[self.actual]

    def addColor(self, color):
        """Adds a color to the list of used colors"""
        if color not in self.used:
            self.used.add(color)
            self.clases = self.clases + 1

    def getUsed(self):
        """Returns the set of colors used"""
        return self.used

    def getClass(self):
        """Returns the actual class, in which we are adding points"""
        return self.actual


class LoadPoints:
    """We introduce bidimensional points of several classes in order to train classify algorithms"""

    def __init__(self):
        """Create the plot figure and configuress the conections"""
        self.fig = plt.figure()
        self.cuadro = self.fig.add_subplot(111)
        self.cuadro.axis([-20, 20, -20, 20])

        self.points = []
        self.classes = []

        self.colors = Color()

        self.cidButton = self.fig.canvas.mpl_connect(
            'button_press_event', self.onclick)
        self.cidKey = self.fig.canvas.mpl_connect(
            'key_press_event', self.on_key)

        plt.show()

    def onclick(self, event):
        """Activates when a new point is added or the color
        has to be changed"""
        # Load the actual color (indicates the class of the point)
        actual_color = self.colors.getColor()
        # In this case, a new point has been added
        if event.button == 1:

            # Plots the received point
            self.cuadro.plot(event.xdata, event.ydata, actual_color)
            self.cuadro.figure.canvas.draw()

            # Add the new point to the list, and its class to the class list
            self.points.append([event.xdata, event.ydata])
            self.classes.append(self.colors.getClass())

            # Checks if there is a new color that has been used
            if actual_color not in self.colors.getUsed():
                self.colors.addColor(actual_color)

        # In this case we have to change the color, that is the class
        elif event.button == 3:

            self.colors.changeColor()

    def on_key(self, event):
        """When the keyboard is pressed we stop adding points"""
        if event.key == 'enter':
            plt.close()

    def get_points(self):
        """Returns an arrray with the points and another with the class to which each belongs"""
        return self.points, self.classes
