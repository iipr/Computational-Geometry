# -*- coding: utf-8 -*-
"""
Created on Sat May 7 12:51:23 2016

@author: Jagui, Iker & Kike
"""


from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import bezier as bz


class Interactive_Bezier():
    def __init__(self):
        self.figure, self.axes = plt.subplots(figsize=(10, 10))
        self.axes.set_xlim(-10, 10)
        self.axes.set_ylim(-10, 10)
        self.touched_circle = None
        self.circles1 = []
        self.circles2 = []
        self.line1 = None
        self.line2 = None
        self.curve1 = None
        self.curve2 = None
        self.bezier1 = None
        self.bezier2 = None
        self.cPoints1 = None
        self.cPoints2 = None
        self.curveindex = 1
        self.index = -1 # indice del circulo tocado
        self.n1 = -1 # numero de puntos del poligono de control de la primera curva menos 1
        self.n2 = -1 # numero de puntos del poligono de control de la segunda curva menos 1
        self.cid_press = self.figure.canvas.mpl_connect('button_press_event', self.click_event)
        self.cid_move = self.figure.canvas.mpl_connect('motion_notify_event', self.motion_event)
        self.cid_release = self.figure.canvas.mpl_connect('button_release_event', self.release_event)
        self.cid_close = self.figure.canvas.mpl_connect('close_event', self.close_event)
        self.figure.canvas.set_window_title('Interactive Bezier curves and intersection')

        # Independent window for selecting the algorithm that computes the curves
        self.algorithm = 'direct'
        self.figButtons = plt.figure(figsize=(2.2,2.2))
        self.figButtons.add_subplot(111)
        self.checkAlgorithm = RadioButtons(self.figButtons.axes[0],('Direct', 'Recursive', 'Horner', 'DeCasteljau', \
                                                                'Subdivision', 'Backward'), activecolor='g')
        self.checkAlgorithm.on_clicked(self.radio_click)
        self.cid_close2 = self.figButtons.canvas.mpl_connect('close_event', self.close_event)
        self.checkAlgorithm.canvas.set_window_title('Algorithm selector')

    def close_event(self, event):
        print('Bye, bye world!')
        self.figure.canvas.mpl_disconnect(self.cid_press)
        self.figure.canvas.mpl_disconnect(self.cid_move)
        self.figure.canvas.mpl_disconnect(self.cid_release)
        self.figure.canvas.mpl_disconnect(self.close_event)
        self.figButtons.canvas.mpl_disconnect(self.cid_close2)

#        plt.close()
        plt.close(self.figure)
        plt.close(self.figButtons)
    
    def radio_click(self, label):
        # Redraw depending on the selected method
        if label == 'Direct':
            self.algorithm = 'direct'
            print 'Direct mode'
        elif label == 'Recursive':
            print 'Recursive mode'
            self.algorithm = 'recursive'
        elif label == 'Horner':
            print 'Horner mode'
            self.algorithm = 'horner'
        elif label == 'DeCasteljau':
            print 'DeCasteljau algorithm'
            self.algorithm = 'deCasteljau'
        elif label == 'Subdivision':
            print 'Subdivision algorithm'
            self.algorithm = 'subdivision'
        elif label == 'Backward':
            print 'Backward differences mode'
            self.algorithm = 'backward'
        self.computeCurve()
        self.drawBezier()
        self.figure.canvas.draw()
#        plt.draw()

    def calculateIndex(self):
        for i in range(self.n1 + 1):
#            print self.cPoints1[i][0], self.cPoints1[i][1]
            if (self.cPoints1[i][0] == self.x0 and self.cPoints1[i][1] == self.y0): 
                self.index = i
                self.curveindex = 1

        for i in range(self.n2 + 1):
#            print self.cPoints2[i][0], self.cPoints2[i][1]
            if (self.cPoints2[i][0] == self.x0 and self.cPoints2[i][1] == self.y0): 
                self.index = i
                self.curveindex = 2

    def computeCurve(self):
        if self.curveindex == 1:
            if self.cPoints1 == None:
                return
            self.n1 = np.size(self.cPoints1, 0) - 1
            if self.algorithm in ['direct', 'recursive', 'horner', 'deCasteljau']:
                num_points = 1000
                self.curve1 = bz.polyeval_bezier(self.cPoints1, num_points, self.algorithm)
            elif self.algorithm == 'subdivision':
                k = 5
                epsilon = 0.01
                self.curve1 = bz.bezier_subdivision(self.cPoints1, k, epsilon)
            elif self.algorithm == 'backward':
                h = 0.01
                m = 100
                self.curve1 = bz.backward_differences_bezier(self.cPoints1, m, h)
        else:
            if self.cPoints1 == None:
                return
            self.n2 = np.size(self.cPoints2, 0) - 1
            if self.algorithm in ['direct', 'recursive', 'horner', 'deCasteljau']:
                self.curve2 = bz.polyeval_bezier(self.cPoints2, 1000, self.algorithm)
            elif self.algorithm == 'subdivision':
                k = 5
                epsilon = 0.01
                self.curve2 = bz.bezier_subdivision(self.cPoints2, k, epsilon)
            elif self.algorithm == 'backward':
                h = 0.01
                m = 100
                self.curve2 = bz.backward_differences_bezier(self.cPoints2, m, h)

    def swapIndex(self):
        if self.curveindex == 1:
            self.curveindex = 2
        else: # curveindex == 2
            self.curveindex = 1

    def computeControlPolygon(self, event):
        if self.curveindex == 1:
            c = plt.Circle((event.xdata, event.ydata), radius=0.2, color='lightblue')
            if self.cPoints1 == None:
                self.cPoints1 = np.asarray([[event.xdata, event.ydata]])
            else:
                self.cPoints1 = np.append(self.cPoints1, [[event.xdata, event.ydata]], axis=0)
            # Now we can compute the curve given cPoints1
            self.computeCurve()
            # Add the (graphic) circle to the list circles2
            self.circles1.append(c)
        else:
            c = plt.Circle((event.xdata, event.ydata), radius=0.2, color='lightgreen')
            if self.cPoints2 == None:
                self.cPoints2 = np.asarray([[event.xdata, event.ydata]])
            else:
                self.cPoints2 = np.append(self.cPoints2, [[event.xdata, event.ydata]], axis=0)
            # Now we can compute the curve given cPoints2
            self.computeCurve()
            # Add the (graphic) circle to the list circles2
            self.circles2.append(c)
        self.axes.add_artist(c)

    def drawBezier(self):
        if self.bezier1 == None and self.n1 == 2:
            self.bezier1 = plt.Line2D(self.curve1[:, 0], self.curve1[:, 1], color='blue')
            self.axes.add_line(self.bezier1)
        elif self.n1 > 2:
            self.bezier1.set_data(self.curve1[:, 0], self.curve1[:, 1])
         
        if self.bezier2 == None and self.n2 == 2:
            self.bezier2 = plt.Line2D(self.curve2[:, 0], self.curve2[:, 1], color='green')
            self.axes.add_line(self.bezier2)
        elif self.n2 > 2:
            self.bezier2.set_data(self.curve2[:, 0], self.curve2[:, 1])

    def drawControlPolygon(self):
        if len(self.circles1) > 0:
            if self.line1 == None:
                self.line1 = plt.Line2D(*zip(*map(lambda x: x.center, self.circles1)), color='lightblue')
                self.axes.add_line(self.line1)
            else:
                self.line1.set_data(*zip(*map(lambda x: x.center, self.circles1)))

        if len(self.circles2) > 0:
            if self.line2 == None:
                self.line2 = plt.Line2D(*zip(*map(lambda x: x.center, self.circles2)), color='lightgreen')
                self.axes.add_line(self.line2)
            else:
                self.line2.set_data(*zip(*map(lambda x: x.center, self.circles2)))

    def click_event(self, event):
        # If we click outside the axes, nothing happens
        if event.inaxes != self.axes: 
            return

        # If we click on the right button, we change from one curve to the other
        # in order to add more control points (modify control points is independent of this)
        if event.button == 3:
            self.swapIndex()
            return

        # For convenience, we allow the user to close the program by clicking with the scroll button
        elif event.button == 2:
            self.close_event(event)
            return
        
        # If we clicked with the left button, we save the event
        self.initial_event = event
        # Check if we have clicked on one of the control points
        for c in self.axes.artists:
            if c.contains(event)[0]:
                self.touched_circle = c
                self.x0, self.y0 = c.center
                # Compute the index of the touched circle in the corresponding cPoints array
                self.calculateIndex() 
                return

        # Create and draw the circle and add it to the corresponding cPoints array
        self.computeControlPolygon(event)

        # If the number of points is at least one, we draw the control polygon
        self.drawControlPolygon()

        # If the number of points is 3 we can create the (graphic) bezier curve
        # If it is greater than 3, we simply recalculate it
        self.drawBezier()

        self.figure.canvas.draw()
        
    def motion_event(self, event):
        if self.touched_circle == None:
            return
        if event.inaxes == self.axes:
            dx = event.xdata - self.initial_event.xdata
            dy = event.ydata - self.initial_event.ydata
            self.touched_circle.center = self.x0 + dx, self.y0 + dy
            if self.curveindex == 1:
                self.line1.set_data(*zip(*map(lambda x: x.center, self.circles1)))
            else:
                self.line2.set_data(*zip(*map(lambda x: x.center, self.circles2)))
                
            self.figure.canvas.draw()
        
    def release_event(self, event):
        if self.touched_circle == None:
            return
        if self.curveindex == 1:
            self.cPoints1[self.index] = np.asarray(self.touched_circle.center)
            if self.n1 > 1:
                self.computeCurve()
                self.bezier1.set_data(self.curve1[:, 0], self.curve1[:, 1])
        else:
            self.cPoints2[self.index] = np.asarray(self.touched_circle.center)
            if self.n2 > 1:
                self.computeCurve()
                self.bezier2.set_data(self.curve2[:, 0], self.curve2[:, 1])
            
        self.touched_circle = None
        self.figure.canvas.draw()
        
if __name__ == "__main__" :
    interactive_Bezier = Interactive_Bezier()
    plt.show()
