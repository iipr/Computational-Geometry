#http://problem-g.estad.ucm.es/FLOP/http//compiler?problemName=GeometriaComputacional/bezier

from __future__ import division
import numpy as np

BINOMIAL_DICT = dict()
RECURSIVE_BERNSTEIN_DICT = dict()

#def polyeval_bezier(P, num_points, algorithm):
'''
    P np.array de dimensión (num_points, dim).
    num_points el numero de puntos en que se divide el intervalo [0, 1].
    algorithm es:
       'direct' -> evaluación directa de los polinomios de Bernstein
       'recursive' -> los polinomios de Bernstein se calculen usando la fórmula recursiva que los caracteriza 
       'horner' ->  Horner para evaluar, dividiendo los valores en los menores que 0.5 y los mayores o iguales a 0.5
       'deCasteljau' -> algoritmo de De Casteljau
    
    Devolverá un np.array de dimensión (num_points, dim) con los valores de la curva de Bézier en los 
    instantes dados por num_points valores equiespaciados en [0, 1].
    '''
    
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
    t_1, t_2 = np.split(t_array,2)
    combarray = np.zeros(n+1)
    for i in range (n+1):
        combarray[i] = comb(n,i)
    coeffs_1 = P*combarray
    horner_1 = np.polyval(coeffs_1, t_1/1-t_1)*((1-t)**n)
    coeffs_2 = P[::1]*combarray
    horner_2 = np.polyval(coeffs_2, 1 -t_1/t_1)*(t**n)
    
    return np.concatenate(horner_1, horner_2)
    
def polyeval_bezier(P, num_points, algorithm):
    t_array = np.linspace(0, 1, num_points)
    if (algorithm == 'direct'):
        return comb(n, k) * t_array**k * (1 - t_array)**(n-k)
    else if (algorithm == 'recursive'):
        bernstein_rec(n, k, t_array)
    else if (algorithm == 'horner'):
        pass
    else if (algorithm == 'deCasteljau'):
        return deCasteljau(k, i, P, t_array)

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