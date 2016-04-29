"""Linear Classifation module

This module provides methods for obtaining Bezier curves 


"""
import numpy as np

BINOMIAL_DICT = dict()
RECURSIVE_BERNSTEIN_DICT = dict()

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
    t = np.linspace(0, 1, num_points)
    P = np.array(P)
    n = np.size(P, 0) - 1
    dim = np.size(P, 1)
    t_array = np.linspace(0, 1, num_points)
    P_axis = np.asarray([P[:, i] for i in range(dim)])
    if algorithm == 'direct':
        return direct(P, t_array)
    elif algorithm == 'recursive':
        bezier = [np.sum(P[k][0] * bernstein_rec(n, k, t_array) for k in range(n+1))]
        for i in range(1, dim):
            bezier = np.concatenate((bezier, [np.sum(P[k][i] * bernstein_rec(n, k, t_array) for k in range(n+1))]))
        return bezier.T
    elif algorithm == 'horner':
        return horner(P, t_array)
    else:
        return deCasteljau_bezier(P, t)

		
def deCasteljau_bezier(P, t):

    # Numero de puntos de P = n+1:
    n, dim = np.shape(P) - np.array([1,0])
    N = np.shape(t)[0]
    
    matrix = np.zeros((N, n+1, dim))
    
    #Inicializacion con el poligono de control
    matrix[:,1:] = t[:,np.newaxis, np.newaxis]*P[1:,:] + (1-t)[:,np.newaxis, np.newaxis]*P[:n,:]
    
    for i in range(2,n+1):
        matrix[:,i:] = t[:,np.newaxis, np.newaxis]*matrix[:,i:] + (1-t)[:,np.newaxis, np.newaxis]*matrix[:,(i-1):n]

    return matrix[:,n]
	
def horner(cPoints, t_array):
    N = t_array.shape[0]
    n, dim = cPoints.shape - np.array([1, 0])
    N0 = int(N / 2)
    t_1 = t_array[:N0]
    t_2 = t_array[N0:]
    #Calculamos los coeficientes para el método de Horner cuando t<1/2
    coeffs_1 = np.asarray([comb_2(n, i) * cPoints[n - i, :] for i in range(n + 1)])
    #Aplicamos Horner y multiplicamos por (1-t)^n
    horner_1 = [np.polyval(coeffs_1[:, j], t_1 / (1 - t_1)) * (1 - t_1)**n for j in range(dim)]

    #Calculamos los coeficientes para el método de Horner cuando t>1/2
    coeffs_2 = np.asarray([comb_2(n, i) * cPoints[i, :] for i in range(n+1)])
    #Aplicamos Horner y multiplicamos por t^n
    horner_2 = [np.polyval(coeffs_2[:, j], (1 - t_2) / t_2) * t_2**n for j in range(dim)]

    return np.hstack((horner_1, horner_2)).T
		
def direct(cPoints, t_array):    
    n = np.size(cPoints, 0) - 1
    return np.sum(np.outer(cPoints[i].T, comb_2(n, i) * t_array ** i * (1 - t_array) ** (n - i)) for i in range(n + 1)).T

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


def bezier_subdivision(P, k, epsilon, lines=False):
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
    n, dim = P.shape - np.array([1, 0])

    if n == 1:
        return P

    delta2_b = np.diff(P, n=2, axis=0)
    norm_delta2 = np.max(np.linalg.norm(delta2_b, axis=1))

    if lines and n*(n - 1) / 8 * norm_delta2 < epsilon:
        return np.array([P[0], P[-1]])

    if k==0 or norm_delta2 < epsilon:
        return P

    P0, P1 = deCasteljau_2(P)
    return np.vstack((bezier_subdivision(P0, k-1, epsilon)[:-1, :], bezier_subdivision(P1, k-1, epsilon)))

	
def deCasteljau_2(P): 
    n = P.shape[0]-1
    dim = P.shape[1]
    bij = np.empty((n+1,n+1, dim))
    bij[0,:,:] = P
    for i in xrange (1, n+1):
        bij[i, :,:] = bij[i-1, :,:]*0.5 +  0.5*np.vstack((np.zeros((1, dim)), bij[i - 1, :n, :]))
    return np.diagonal(bij).T, bij[:,n][::-1]


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