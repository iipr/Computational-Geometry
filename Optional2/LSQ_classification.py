import numpy as np
from numpy import linalg as la


class LSQ_classification:

    def __init__(self):

        # Initialize W for painting
        self.W = None

    def compute_W(self, X, tags, K):
        '''Computes the matrix of coefficients of the K
        affine forms used for the linear square clasification
        of points as explained in Bishop, ch.4.1

        X: list of training points
            [[x0, y0], ...]
        tags: list of classes to which the data points belong
            [k0, k1, ...]
        K: number of different classes'''
        # Convert to array type
        X = np.asarray(X)

        
        # Compute the number of points given
        N = X.shape[1]
        # Added ones to X_tilde(Bishop)
        X_tilde = np.vstack((np.ones(N), X))
        T = np.zeros((N, K))
        # If the point belongs to the class k, it is a 1.
        # Otherwise, we set a 0
        # We do this for every point, generating T matrix (see Bishop)
        for row, tag in enumerate(tags):
            T[row, tag] = 1


        # The function solves the least squares system (it can be used for checking):
        # self.W = np.linalg.lstsq(X_tilde.transpose(), T)[0]

        # We compute the pseudoinverse of X_tilde
        pseudo = np.linalg.inv(X_tilde.dot(X_tilde.T))

        # We obtain matrix W of classification by least squares
        self.W = np.dot(pseudo.dot(X_tilde), T)

    def classifier(self, points):

        '''Classifies the points given in the classes,
        using the W obtained with the training points'''
        if self.W is None:
            print 'First train the classifier!'
            return
        points = np.asarray(points)
        M = points.shape[1]
        # Added ones to fill X_tilde
        pts_tilde = np.vstack((np.ones(M), points))

        # According to Bishop, we obtain the classification
        # matrix. We return the highest value k of y_k for
        # each point.
        Y = (self.W.transpose()).dot(pts_tilde)
        return np.argmax(Y, axis=0)
