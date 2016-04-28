#http://problem-g.estad.ucm.es/FLOP/http//compiler?problemName=GeometriaComputacional/bezier

from __future__ import division
import numpy as np

BINOMIAL_DICT = dict()
RECURSIVE_BERNSTEIN_DICT = dict()

#Para pruebas en el main:
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time


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
    P = np.asarray(P)
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
        return bezier.T[::-1]

    elif(algorithm == 'deCasteljau'):
        bezier = [deCasteljau(n, 0, P_axis[0, :], t_array)]
        for i in range(1, dim):
            bezier = np.concatenate((bezier, [deCasteljau(n, 0, P_axis[i, :], t_array)]))
        return bezier.T[::-1]

def bezier_subdivision_recursive(P, k, epsilon, lines):
    #Calcular max (viene en los apuntes)
    if(k==0 or max < epsilon):
        if(lines):
			#solo los extremos
            pass
        else:
            pass
        #dibujar b
    else:
        #Calcular a0,...,a2n sobre [0, 0.5, 1]
		#Calc a0,...,an
        a_1 = bezier_subdivision_recursive(P, k-1, epsilon, lines)
        #Calc an,...,a2n
        a_2 = bezier_subdivision_recursive(P, k-1, epsilon, lines)
        return a_1,a_2

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


if __name__ == '__main__':
    fig = plt.figure()
    ax = Axes3D(fig)

    P = np.asarray([[-95, 98, -74], [-46, -52, 33], [76, 3, 96], [-25, 72, -98], [4, 95, 75], [77, -27, 98], [65, 65, 19], [87, -50, 8], [-68, -11, -84], [6, 3, 16], [-38, 4, 76], [-77, -55, 72], [89, 36, 83]])
    num_points = 100
    dim = np.size(P, 1)
    n = np.size(P, 0) - 1
    algorithm = 'deCasteljau'
    plt.title('De Casteljau: Prueba 3D')
    start = time.time()
    result = polyeval_bezier(P, num_points, algorithm)
#    print result
#    print np.size(result)
    ax.plot(result[0, :], result[1, :], result[2, :])
    P_x = P[:, 0]
    P_y = P[:, 1]
    P_z = P[:, 2]
    ax.plot(P_x, P_y, P_z, 'ro')
    ax.plot(P_x, P_y, P_z, 'g')
    end = time.time()
    plt.show()
    print(end - start)
