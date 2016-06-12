from __future__ import division
import numpy as np

from sympy import solve, symbols, Eq

def degenerateCases( p1, p2, p3, p4 ):
    """Si en nuestra ecuacion la x o la y 
    son alguna cero, solo tenemos una incognita"""
    if ( ( p2[0] - p1[0] ) == 0 ) and ( ( p2[1] - p1[1] ) == 0 ) and \
       ( ( p4[0] - p3[0] ) == 0 ) and ( (p4[1] - p3[1] ) == 0 ):
        """Tenemos dos puntos, si son los mismos 
           lo devolvemos, si no no hacemos nada."""

        if ( p1[0] == p3[0] ) and ( p1[1] == p3[1] ):
            return p1
        else:
            return None
    if (p2[0]-p1[0])==0 and (p4[0]-p3[0])==0:
        """Dos segmentos verticales"""
        if p1[0]-p3[0]==0:
            """Segmentos en la misma vertical"""
            if max(p1[1], p2[1]) <= max(p3[1], p4[1]) and max(p1[1], p2[1]) >= min(p3[1], p4[1]):
                return  [p1[0], max(p1[1], p2[1])]
            elif min(p1[1], p2[1]) <= max(p3[1], p4[1]) and min(p1[1], p2[1]) >= min(p3[1], p4[1]):
                return  [p1[0], min(p1[1], p2[1])]
            else:
                return None
        else:
            return None
        
    if (p2[1]-p1[1])==0 and (p4[1]-p3[1])==0:
        """Dos segmentos horizontales"""
        if p1[1]-p3[1]==0:
            """Segmentos en la misma horizontal"""
            if max(p1[0], p2[0]) <= max(p3[0], p4[0]) and max(p1[0], p2[0]) >= min(p3[0], p4[0]):
                return  [max(p1[0], p2[0]), p1[1]]
            if min(p1[0], p2[0]) <= max(p3[0], p4[0]) and min(p1[0], p2[0]) >= min(p3[0], p4[0]):
                return  [min(p1[0], p2[0]), p1[1]]
            else:
                return None
        else:
            return None

    
def intersectSegments(p1, p2, p3, p4):
    """Recibe cuatro puntos, los dos primeros son 
    un segmento y los dos ultimos son otro,
    devuelve la interseccion de ambos o None si no hay.
    En algunosl casos degenerados devolvemos None."""
    
    if ((p2[0]-p1[0])==0 and (p4[0]-p3[0])==0)or((p2[1]-p1[1])==0 and (p4[1]-p3[1])==0):
        """Casos degenerados: Dos rectas verticales, dos horizontales o cuatro puntos"""
        return degenerateCases(p1, p2, p3, p4)
    
    
    
    #Ecuacion para la interseccion de dos rectas en R2
    #x, y = symbols('x y')
    #sol = solve([Eq(p1[0] + x*(p2[0]-p1[0]) - p3[0] - y*(p4[0]-p3[0]),0), 
    #           Eq(p1[1] + x*(p2[1]-p1[1]) - p3[1] - y*(p4[1]-p3[1]), 0)], [x, y])

    # Utilizamos la solucion manual nuestra del sistema.
    a = p2[0] - p1[0]
    b = p4[0] - p3[0]
    c = p3[0] - p1[0]
    d = p2[1] - p1[1]
    e = p4[1] - p3[1]
    f = p3[1] - p1[1]

    if ( b * d - a * e ) == 0 or a == 0:
         return None

    mu = ( f * a - d * c ) / ( b * d - a * e )
    lam = ( c + mu * b ) / a 


    if ( lam <= 1 ) and ( lam >= 0 ) and ( mu <= 1 ) and ( mu >= 0 ): 
        """Como tomamos p1 y el vector p1-p2, el lambda de la interseccion
        debe estar entre 0 y 1, es decir dentro del segmento"""
        return np.array( p1 + lam * ( p2 - p1 ) )
    else:
        return None

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib.lines as ln
    import time as t

    fig = plt.figure()

    sub = fig.add_subplot(111)
    p1 = np.array([5,5])
    p2 = np.array([2,2])

    p3 = np.array([1,6])
    p4 = np.array([7,1])

    line1 = ln.Line2D(np.array([p1[0], p2[0]]), np.array([p1[1], p2[1]]), color='b')
    line2 = ln.Line2D(np.array([p3[0], p4[0]]), np.array([p3[1], p4[1]]), color='b')

    sub.add_line(line1)
    sub.add_line(line2)
    sub.axis([0, 7, 0, 7])

    t0 = t.time()
    inter = intersection(p1, p2, p3, p4)
    print t.time() - t0
    if inter == None:
        print 'no hay inters'
    else:
        sub.plot(float(inter[0]), float(inter[1]), 'ro')

    sub.figure.canvas.draw()
    plt.show()
