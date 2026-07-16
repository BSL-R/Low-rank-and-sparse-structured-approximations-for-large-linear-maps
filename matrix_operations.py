import numpy as np

def svd(A):

    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)

    return U, S, Vt

# different depending on dimensions of array inputted
# a,b are 1D arrays (column vector if left multiplying, row vector if right multiplying) 
# A is 2D array (matrix) 

def left_multiply(A, a):

    A = np.array(A)
    a = np.array(a)

    return np.dot(A, a)

def right_multiply(a, A):

    a = np.array(a)
    A = np.array(A)

    return np.dot(a, A)

def dot_product(a, b):

    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b)