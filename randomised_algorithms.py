import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import SVD_investigation as svd


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



def svd_error_random(A, r, q, norm_type): 

    A_randsvd, U, S, V_t = svd_random(A, r, q) 
    error = np.linalg.norm(A_randsvd - A, norm_type) 

    return error



def error_ranksize_table_random(r_max, n_max, q, norm_type, dist, **dist_kwargs):

    rank_list = np.arange(2, r_max+1, 2)
    size_list = [f"{i}x{i}" for i in range(1, n_max+1, 1)]
    matrix_list = [dist(size=(i, i), **dist_kwargs) for i in range(1, n_max+1)]

    rankr_error_table = np.array([ [svd_error_random(matrix_list[j], rank_list[i], q, norm_type) for j in range(n_max)] for i in range(r_max//2)])

    table = pd.DataFrame(rankr_error_table, index=rank_list, columns=size_list)
    print(table)

    plt.figure(figsize=(12, 6))
    plt.imshow(table, cmap="Reds", aspect="auto") 
    plt.colorbar(label="Error")

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            plt.text(j, i, f"{table.iloc[i, j]:.2f}",
                    ha="center", va="center")

    plt.xticks(np.arange(len(table.columns)), table.columns, rotation=45)
    plt.yticks(np.arange(len(table.index)), table.index)

    plt.xlabel("matrix size")
    plt.ylabel("rank")
    plt.title("Rank-r approximation error")
    plt.show()



def error_ranksize_plot_random(r, n_max, q, norm_type, dist, **dist_kwargs):

    matrix_list = [dist(size=(i, i), **dist_kwargs) for i in range(1, n_max+1)]
    rankr_error_list = np.array([svd_error_random(matrix_list[j], r, q, norm_type) for j in range(n_max)])
    
    index_list = np.arange(1, n_max+1, 1)

    plt.scatter(index_list, rankr_error_list, color='red', s=20)
    plt.xlabel("#rows of the square matrix")
    plt.ylabel("Error in approx.")
    plt.title("Error against size (fixed rank)")
    plt.show()



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