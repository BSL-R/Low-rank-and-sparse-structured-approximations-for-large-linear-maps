import numpy as np

def svd(A):

    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)

    return U, S, Vt


def left_multiply(A, x):

    A = np.array(A)
    x = np.array(x)

    return np.dot(A, x)

