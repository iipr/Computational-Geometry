# -*- coding: utf-8 -*-

"""Linear Classifation module

This module provides methods for obtaining Bezier curves 


"""
from __future__ import division
import numpy as np


BINOMIAL_DICT = dict()
RECURSIVE_BERNSTEIN_DICT = dict()

def comb(m, n):
    if(not((m, n) in BINOMIAL_DICT)):
        BINOMIAL_DICT[(m, n)] = np.math.factorial(m) // (np.math.factorial(n) * np.math.factorial(m - n))
    return BINOMIAL_DICT.get((m, n))


def comb_2(m, n):
    if((m, n) in BINOMIAL_DICT):
        return BINOMIAL_DICT.get((m, n))
    elif(m == n or n == 0):
        BINOMIAL_DICT[(m, n)] = 1
        return 1
    else:
        BINOMIAL_DICT[(m, n)] = comb_2(m - 1, n - 1) + comb_2(m - 1, n)
        return BINOMIAL_DICT.get((m, n))

def bernstein_rec(n, k, t):
    if((n, k) in RECURSIVE_BERNSTEIN_DICT):
        return RECURSIVE_BERNSTEIN_DICT.get((n, k))
    elif(k == -1 or k == n + 1):
        RECURSIVE_BERNSTEIN_DICT[(n, k)] = 0
        return 0
    elif(n == 0 and k == 0):
        RECURSIVE_BERNSTEIN_DICT[(n, k)] = 1
        return 1
    else:
        RECURSIVE_BERNSTEIN_DICT[(n, k)] = t * bernstein_rec(n - 1, k - 1, t) + (1 - t) * bernstein_rec(n - 1, k, t)
        return RECURSIVE_BERNSTEIN_DICT.get((n, k))

def deCasteljau(k, i, cp, t):
    if(k == 0):
        return cp[i]
    return deCasteljau(k - 1, i, cp, t) * t + deCasteljau(k - 1, i + 1, cp, t) * (1 - t)

def horner(cPoints, t_array):
    n, dim = cPoints.shape - np.array([1, 0])
    #Partimos en dos partes: <1/2 y >1/2
    t_1, t_2 = np.split(t_array, 2)

    #Calculamos los coeficientes para el método de Horner cuando t<1/2
    coeffs_1 = np.asarray([comb_2(n, i) * cPoints[n - i, :] for i in range(n + 1)])
    #Aplicamos Horner y multiplicamos por (1-t)^n
    horner_1 = [np.polyval(coeffs_1[:, j], t_1 / (1 - t_1)) * (1 - t_1)**n for j in range(dim)]

    #Calculamos los coeficientes para el método de Horner cuando t>1/2
    coeffs_2 = np.asarray([comb_2(n, i) * cPoints[i, :] for i in range(n+1)])
    #Aplicamos Horner y multiplicamos por t^n
    horner_2 = [np.polyval(coeffs_2[:, j], (1 - t_2) / t_2) * t_2**n for j in range(dim)]

    return np.hstack((horner_1, horner_2)).T

def polyeval_bezier(P, num_points, algorithm):
    """
    This function returns an array with the values of Bezier's curve evaluated
    in the points given by the parameter 'numpoints'

    The option 'direct' will force a direct evaluation of Bernstein's polynomials,
    'recursive' will calculate Bernstein's polynomials using its recursive formulae,
    'horner' will use Horner's method in the evaluation of these polynomials, and finally 
    'deCasteljau' will evaluate the curve using De Casteljau's Algorithm.

    Example
    -------
    P = [[-39, -71, -54], [82, -39, 1], [-14, -89, -61], [-50, 25, -78], [26, 40, 51], [-43, -81, 78]
    polyeval_bezier(P, 100, 'horner')
    Q = [[0.1, .8], [1,.2], [0,.1], [.3,.9]]
    polyeval_bezier(Q, 50, 'direct')
	"""
    n = np.size(P, 0) - 1
    dim = np.size(P, 1)
    t_array = np.linspace(0, 1, num_points)
    P_axis = np.asarray([P[:, i] for i in range(dim)])
    if(algorithm == 'direct'):
        bezier = [np.sum(P[k][0] * comb(n, k) * t_array** k * (1 - t_array) ** (n - k) for k in range(n+1))]
        for i in range(1, dim):
            bezier = np.concatenate((bezier, [np.sum(P[k][i] * comb(n, k) * t_array ** k * (1 - t_array) ** (n - k) for k in range(n+1))]))
        return bezier.T

    elif(algorithm == 'recursive'):
        bezier = [np.sum(P[k][0] * bernstein_rec(n, k, t_array) for k in range(n+1))]
        for i in range(1, dim):
            bezier = np.concatenate((bezier, [np.sum(P[k][i] * bernstein_rec(n, k, t_array) for k in range(n+1))]))
        return bezier.T

    elif(algorithm == 'horner'):
        return horner(P, t_array)

    elif(algorithm == 'deCasteljau'):
        bezier = [deCasteljau(n, 0, P_axis[0, :], t_array)]
        for i in range(1, dim):
            bezier = np.concatenate((bezier, [deCasteljau(n, 0, P_axis[i, :], t_array)]))
        return bezier.T[::-1]


def bezier_subdivision(P, k, epsilon, lines): 
    """
    This function implements subdivision's method
    Integer parameter k indicates the number of subdivision, and epsilon
    is the stop threshold, which measures how close to a straight line is the curve.

    The function will return an array containing the sequence of points given by the resulting 
    Bezier polygons. 

    If lines = True, it will only return the succesion of exterms, wih no intermediate points.

    Example
    -------
    num_points = 100
    epsilon = 0.01
    k = 10
    P = [[21, -17, -27], [92, -46, -36], [-14, -66, -75], [72, -41, -89], [49, -37, 83], [-48, 63, 66], [-94, -51, 71]]
    result = bezier_subdivision(np.array(P), k, epsilon, True)

    """
    P = np.array(P)
    n = np.shape(P)[0]
    #Stop threshold calculation
    delta2_b = np.diff(P, n=2, axis=0)
    threshold = np.max(np.linalg.norm(delta2_b, axis=1))
    
    if lines and n*(n - 1) / 8 * threshold < epsilon:
        return np.array([P[0], P[-1]])

    if k==0 or threshold < epsilon:
        return P

	#Division of the problem in two subproblems
    P_1, P_2 = deCasteljau_2(P)
	#We have to eliminate the last point, as it is duplicated in both arrays
    a_1 = bezier_subdivision(P_1, k-1, epsilon, lines)[:-1, :]
    a_2 = bezier_subdivision(P_2, k-1, epsilon, lines)
    return np.vstack((a_1, a_2))

	
def deCasteljau_2(P): 
    P = np.array(P)
    n = P.shape[0]-1
    dim = P.shape[1]
    bij = np.empty((n+1,n+1, dim))
    b_diag = np.empty((n+1, dim))
    bij[0,:,:] = P
    b_diag[0] = P[0]
    for i in range (1, n+1):
        for j in range (i, n+1):
            bij[i, j,:] = bij[i-1, j,:]*0.5 + bij[i-1, j-1,:]*0.5
            if (i == j):
                b_diag[j] = bij[i,j]
    return b_diag, bij[:,n][::-1]

	
def backward_differences_bezier(P, m, h=None):
    """
    This function will evaluate Bezier's curve at points of the form h * k for k = 0, ..., m.
    If h = None then h = 1/m. 
    The function uses the method of backward differences explained in class.

    Example
    -------
    num_points = 100
    h = 0.05
    m = 100
    P = [[-90, 29, 51], [-32, 80, -15], [-50, -40, -91], [-35, 93, 68], [-58, -97, 21]]
    result = backward_differences_bezier(P, k, epsilon, True)
    
    """

    if h == None:
        h = 1/m
    n = np.shape(P)[0]-1
    d = np.shape(P)[1]

    #Necesitaremos una matriz 'mxnxd' con n grado de la curva
    points = np.zeros((m+1, n+1, d))
    
    #Calculo del triangulo inicial
    t_array = np.arange(0, (n + 1)*h, h)

    points[:(n+1),0] = horner(P, t_array)
 
    #Diferencias hacia delante
    for i in range (1,n+1):
        for j in range (1,i+1):
            points[i, j] = points[i, j-1] - points[i-1, j-1]
    #Completamos la columna constante
    points[(n+1):, n] = points[n,n]
    
    #Calculo final
    for i in range(n+1, m+1):
        for j in range(n-1, -1, -1):
            points[i,j] = points[i, j+1] + points[i-1, j]
    
    #Devolvemos los p0...pM puntos, que estan en la primera columna
    return points[:,0]
