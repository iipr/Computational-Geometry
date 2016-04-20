#http://problem-g.estad.ucm.es/FLOP/http//compiler?problemName=GeometriaComputacional/bezier

from __future__ import division
import numpy as np

def polyeval_bezier(P, num_points, method):
    '''
    P np.array de dimensión (num_points, dim).
    num_points el numero de puntos en que se divide el intervalo [0, 1].
    method (será sustituido por algorith ha dicho) es 'direct', 'recursive', 'horner' o 'deCasteljau'.
    
    Devolverá un np.array de dimensión (num_points, dim) con los valores de la curva de Bézier en los 
    instantes dados por num_points valores equiespaciados en [0, 1].
    '''
    t = np.linspace(0, 1, num_points - 1)

def bezier_subdivision(P, k, epsilon, lines=False):
    '''
    Metodo de subdivision. 
    
    P np.array de dimensión (num_points, dim).
    El entero k indicará el número de subdivisiones.
    Epsilon será el umbral de parada, que mide cuán cercana a una recta está la curva.
    Si lines=True, devolverá sólo la sucesión de extremos, sin puntos intermedios.
    Devolverá un np.array que contendrá la sucesión de puntos dada por los polígonos de Bézier resultantes.
    '''
    pass

def backward_differences_bezier(P, m, h=None):
    '''
    Método de diferencias "hacia atrás".
    
    Evaluará la curva de Bézier en los puntos de la forma h*k para k=0,...,m. Si h=None entonces h=1/m.
    '''
    pass
