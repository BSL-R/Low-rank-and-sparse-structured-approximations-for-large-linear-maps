import numpy as np
import matplotlib.pyplot as plt



def svd_optimal_rankr(A, r):

# unique iff S[r-1]>S[r]
# otherwise any of the rth largest singular values chosen (depending on list S)

    A=np.array(A)
    U,S,Vt = np.linalg.svd(A, full_matrices=False)

    S = np.diag(S)

    S_r = S[ :r, :r]
    U_r = U[ : , :r]
    Vt_r = Vt[ :r, : ]

    A_r = U_r @ S_r @ Vt_r

    return A_r, U_r, S_r, Vt_r



def svd_error_rankr(A, r):

    A_r, U_r, S_r, Vt_r = svd_optimal_rankr(A, r)
    error = np.linalg.norm(A_r - A, 'fro')

    return error


# using normal distribution around optimal solution for perturbated matrices (any rank)
# scaled variance so that perturbations are reasonable, S_r[0,0] is largest singular value

def svd_perturbation_plot(A, r, num_perturbed):

    A_r, U_r, S_r, Vt_r = svd_optimal_rankr(A, r)

# vectorising using 3D numpy array instead of loop
    A_perturbed = A_r + np.random.normal(0, 0.05*S_r[0,0], (num_perturbed, A_r.shape[0], A_r.shape[1]))

# x-axis: distance (Frobenius) of perturbed matrix from optimal rank-r truncation
    distance_approx = np.linalg.norm(A_perturbed - A_r, 'fro', axis=(1,2))
    distance_true = np.linalg.norm(A_perturbed - A, 'fro', axis=(1,2))
    
    plt.scatter(distance_approx, distance_true, color='red', s=20)
    plt.xlabel(f"Frobenius dist. from rank-{r} optimal sol.")
    plt.ylabel("Frobenius dist. from true matrix")
    plt.title("Perturbation Scatter")
    plt.show()

#A = [[14,24,8], [54,12,63],[6,32,19]] 
#svd_perturbation_plot(A,1,1500) 
#svd_perturbation_plot(A,2,1500) 
#svd_perturbation_plot(A,3,1500) 


###########################################################################################################################################################################

import pandas as pd

# for Gaussian square matrices

def error_ranksize_table(r, n):

    rank_list = np.arange(1, r+1, 1)
    size_list = [f"{i}x{i}" for i in range(1, n+1, 1)]
    matrix_list = [np.random.normal(0, 1, (i, i)) for i in range(1, n+1)]

    rankr_error_table = np.array([ [svd_error_rankr(matrix_list[j], rank_list[i]) for j in range(n)] for i in range(r)])

    table = pd.DataFrame(rankr_error_table, index=rank_list, columns=size_list)

    print(table)


def error_ranksize_plot(r, n):

    matrix_list = [np.random.normal(0, 1, (i, i)) for i in range(1, n+1)]
    rankr_error_list = np.array([svd_error_rankr(matrix_list[j], r) for j in range(n)])
    
    index_list = np.arange(1, n+1, 1)

    plt.scatter(index_list, rankr_error_list, color='red', s=20)
    plt.xlabel("#rows of square Guassian matrix")
    plt.ylabel("Error in approx.")
    plt.title("Error against size (fixed rank)")
    plt.show()

#error_ranksize_plot(100,100)
#error_ranksize_plot(200,200)