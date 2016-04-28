#http://problem-g.estad.ucm.es/FLOP/http//compiler?problemName=GeometriaComputacional/bezier

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

def horner(n, cp, t_array):
    #Partimos en dos partes: <1/2 y >1/2
    t_1, t_2 = np.split(t_array,2)

    #Calculamos los coeficientes para el método de Horner cuando t<1/2
    coeffs_1 = [comb_2(n, i) * cp[i] for i in range(n+1)]
    #Aplicamos Horner y multiplicamos por (1-t)^n
    horner_1 = np.polyval(coeffs_1, t_1 / (1 - t_1)) * (1 - t_1)**n
    #Calculamos los coeficientes para el método de Horner cuando t>1/2
    coeffs_2 = [comb_2(n, i) * cp[::-1][i] for i in range(n+1)]
    #Aplicamos Horner y multiplicamos por t^n
    horner_2 = np.polyval(coeffs_2, (1 - t_2) / t_2) * t_2**n

    return np.concatenate([horner_1, horner_2])

def polyeval_bezier(P, num_points, algorithm):
    # Numero de puntos de P = n+1:
    n = np.size(P, 0) - 1
    # Dimension de los puntos
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
        bezier = [horner(n, P_axis[0, :], t_array)]
        for i in range(1, dim):
            bezier = np.concatenate((bezier, [horner(n, P_axis[i, :], t_array)]))
        return bezier

    elif(algorithm == 'deCasteljau'):
        bezier = [deCasteljau(n, 0, P_axis[0, :], t_array)]
        for i in range(1, dim):
            bezier = np.concatenate((bezier, [deCasteljau(n, 0, P_axis[i, :], t_array)]))
        return bezier


def bezier_subdivision(P, k, epsilon, lines): 
    P = np.array(P)
    #calcular max
    n = np.shape(P)[0]
    delta2_b = np.diff(P, n=2, axis=0)
    threshold = np.max(np.linalg.norm(delta2_b, axis=1))

    if lines and n*(n - 1) / 8 * threshold < epsilon:
        return np.array([P[0], P[-1]])

    if k==0 or threshold < epsilon:
        return P

    P_1, P_2 = deCasteljau_2(P)
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
