import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import svd_investigation as svd


# Original matrix A has dim mxn
# Rank-2k approx.

# Stage-A of prototype for randomised SVD 
def stage_A(A, k, q):

    test_matrix = np.random.normal(0, 1, (A.shape[1], 2*k))

    Y = A @ test_matrix  

    # Orthonormalisation step between each multiplication of A and A* (as per note)
    Q, _ = np.linalg.qr(Y, mode='reduced')
    
    for _ in range(q):
       
        Q = A.conj().T @ Q   
        Q, _ = np.linalg.qr(Q, mode='reduced')
        
        Q = A @ Q  
        Q, _ = np.linalg.qr(Q, mode='reduced')

    return Q


# Stage-B of prototype for randomised SVD
def stage_B(A, k, q):

    Q = stage_A(A, k, q)

    B = Q.conj().T @ A
    U_B, S, V_t = np.linalg.svd(B, full_matrices=False)
    S = np.diag(S)

    U = Q @ U_B

    return U, S, V_t

###########################################################################################################################################################################

# Final rank-r factorisation using randomised SVD (r must be integer multiple of 2)
def svd_random(A, r, q):

    if r % 2 != 0:
        return None

    U, S, V_t = stage_B(A, r//2, q)
    A_randsvd = U @ S @ V_t

    return A_randsvd, U, S, V_t



###########################################################################################################################################################################

if __name__ == "__main__": 
# Testing on Gaussian matrices

    A = svd.random_matrix((5,6), np.random.normal, {'loc':0, 'scale':1})

    svd.svd_free_rank_perturbation_plot(A, 2, 1000, 'fro', svd_random, {'q':0})
    svd.svd_free_rank_perturbation_plot(A, 2, 1000, 'fro', svd_random, {'q':1})
    svd.svd_free_rank_perturbation_plot(A, 2, 1000, 'fro', svd_random, {'q':2})

    svd.svd_free_rank_perturbation_plot(A, 4, 1000, 'fro', svd_random, {'q':0})
    svd.svd_free_rank_perturbation_plot(A, 4, 1000, 'fro', svd_random, {'q':1})
    svd.svd_free_rank_perturbation_plot(A, 4, 1000, 'fro', svd_random, {'q':2})

    svd.svd_free_rank_perturbation_plot(A, 6, 1000, 'fro', svd_random, {'q':0})
    svd.svd_free_rank_perturbation_plot(A, 6, 1000, 'fro', svd_random, {'q':1}) 
    svd.svd_free_rank_perturbation_plot(A, 6, 1000, 'fro', svd_random, {'q':2}) 

    svd.error_ranksize_table(5, 5, 'fro', svd_random, {'q':0}, np.random.normal, {'loc':0, 'scale':1}) 
    svd.error_ranksize_table(5, 5, 'fro', svd_random, {'q':1}, np.random.normal, {'loc':0, 'scale':1}) 
    svd.error_ranksize_table(5, 5, 'fro', svd_random, {'q':2}, np.random.normal, {'loc':0, 'scale':1}) 

    svd.error_ranksize_plot(10, 50, 'fro', svd_random, {'q':0}, np.random.normal, {'loc':0, 'scale':1})
    svd.error_ranksize_plot(10, 50, 'fro', svd_random, {'q':1}, np.random.normal, {'loc':0, 'scale':1})
    svd.error_ranksize_plot(10, 50, 'fro', svd_random, {'q':2}, np.random.normal, {'loc':0, 'scale':1})

    svd.svd_fixed_rank_perturbation_plot(A, 2, 'fro', 50, svd_random, {'q':0}, False)
    svd.svd_fixed_rank_perturbation_plot(A, 2, 'fro', 50, svd_random, {'q':1}, False)
    svd.svd_fixed_rank_perturbation_plot(A, 2, 'fro', 50, svd_random, {'q':2}, False)

    svd.svd_fixed_rank_perturbation_plot(A, 2, 'fro', 50, svd_random, {'q':0}, True)
    svd.svd_fixed_rank_perturbation_plot(A, 2, 'fro', 50, svd_random, {'q':1}, True)
    svd.svd_fixed_rank_perturbation_plot(A, 2, 'fro', 50, svd_random, {'q':2}, True)