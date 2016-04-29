import numpy as np

def polyeval_bezier(P, num_points, algorithm):
    t = np.linspace(0, 1, num_points)
    if algorithm == 'direct':
        return direct_bezier(P, t)
    elif algorithm == 'recursive':
        return recursive_eval_bezier(P, t)
    elif algorithm == 'horner':
        return horner_eval_bezier(P, t)
    else:
        return deCasteljau_bezier(P, t)


BINOMIAL_DICT = dict()
def binomial_coefficient(n, k):
    if (n, k) in BINOMIAL_DICT:
        return BINOMIAL_DICT[(n, k)]
    if k == 0 or n == k:
        return 1
    else:
        result = binomial_coefficient(n - 1, k) + binomial_coefficient(n - 1, k - 1)
        BINOMIAL_DICT[(n, k)] = result
        return result


def direct_bezier(P, t):
    n, dim = P.shape - np.array([1, 0])
    t = t[:, np.newaxis]
    one_minus_t = 1 - t
    return sum(binomial_coefficient(n, k) * t**k * (one_minus_t)**(n - k) * P[k]
        for k in range(n + 1))


def horner_eval_bezier(P, t):
    N = t.shape[0]
    n, dim = P.shape - np.array([1, 0])
    N0 = int(N / 2)
    t0 = t[:N0]
    t1 = t[N0:]

    factor0 = np.array([binomial_coefficient(n, k) * P[n - k, :] for k in range(n + 1)])
    factor1 = np.array([binomial_coefficient(n, k) * P[k, :] for k in range(n + 1)])

    onemt0 = 1 - t0
    onemt1 = 1 - t1

    eval_bezier0 = np.array([onemt0**n * np.polyval(factor0[:, d], t0 / onemt0) for d in range(dim)])
    eval_bezier1 = np.array([t1**n * np.polyval(factor1[:, d], onemt1 / t1) for d in range(dim)])

    return np.hstack((eval_bezier0, eval_bezier1)).T


RECURSIVE_BERNSTEIN_DICT = dict()
def recursive_bernstein(n, k, t):
    one_minus_t = 1 - t
    if  (n, k) in RECURSIVE_BERNSTEIN_DICT:
        return RECURSIVE_BERNSTEIN_DICT[(n, k)]
    if (n, k) == (1, 0): return one_minus_t
    if (n, k) == (1, 1): return t
    if k < 0 or n < k: return 0
    RECURSIVE_BERNSTEIN_DICT[(n, k)] = one_minus_t*recursive_bernstein(n - 1, k, t) + t*recursive_bernstein(n - 1, k - 1, t)
    return RECURSIVE_BERNSTEIN_DICT[(n, k)]


def recursive_eval_bezier(P, t):
    n, dim = P.shape - np.array([1, 0])
    return  sum(recursive_bernstein(n, k, t)[:, np.newaxis]*P[k]
        for k in range(n + 1))


def deCasteljau_bezier(P, t):
    num_points = t.shape[0]
    N, dim = P.shape - np.array([1, 0])
    one_minus_t = 1 - t
    P = one_minus_t[:, np.newaxis, np.newaxis]*P[:, :] + t[:, np.newaxis, np.newaxis]*np.vstack((P[1:, :], np.zeros(dim)))
    for i in xrange(2, N + 1):
        P = one_minus_t[:, np.newaxis, np.newaxis]*P + t[:, np.newaxis, np.newaxis]*np.hstack((P[:, 1:, :], np.zeros((num_points, 1, dim))))
    return P[:, 0, :]



def bezier_subdivision(P, k, epsilon, lines=False):
    n, dim = P.shape - np.array([1, 0])

    if n == 1:
        return P

    delta2_b = np.diff(P, n=2, axis=0)
    norm_delta2 = np.max(np.linalg.norm(delta2_b, axis=1))

    if lines and n*(n - 1) / 8 * norm_delta2 < epsilon:
        return np.array([P[0], P[-1]])

    if k==0 or norm_delta2 < epsilon:
        return P

    P0, P1 = subdivide(P)
    return np.vstack((bezier_subdivision(P0, k-1, epsilon)[:-1, :], bezier_subdivision(P1, k-1, epsilon)))



def subdivide(P):
    N, dim = P.shape - np.array([1, 0])
    deCasteljauP = np.empty((N + 1, N + 1, dim))
    deCasteljauP[0, :, :] = P
    for i in xrange(1, N + 1):
        deCasteljauP[i, :, :] = 0.5*deCasteljauP[i - 1, :, :] +\
                                0.5*np.vstack((deCasteljauP[i - 1, 1:, :], np.zeros((1, dim))))

    P0 = deCasteljauP[:, 0, :]
    P1 = np.diagonal(deCasteljauP[:, ::-1, :]).T

    return P0, P1[::-1, :]



def backward_differences_bezier(P, m, h=None):
    if h == None:
        h = 1/m

    n = P.shape[0] - 1
    r = m - n
    tri_matrix = np.tri(r, r, 0)
    t0 = np.arange(0, (n + 1)*h, h)

    p_init = horner_eval_bezier(P, t0).T

    deltas_p = {k:np.diff(p_init, k).T for k in range(n + 1)}

    extended_deltas = dict()
    extended_deltas[n] = np.repeat(deltas_p[n], r, axis=0)

    for k in range(1, n + 1):
        indep_terms = extended_deltas[n - k + 1]
        indep_terms[0] = indep_terms[0] + deltas_p[n - k][k]
        extended_deltas[n - k] = tri_matrix.dot(indep_terms)

    return np.vstack((p_init.T, extended_deltas[0]))
 
