import numpy as np
import matplotlib.pyplot as plt

VARIANCE_FORMULAS = {
    np.random.normal: lambda loc, scale: scale**2,
    np.random.uniform: lambda low, high: (high - low)**2 / 12,
    np.random.poisson: lambda lam: lam,
    np.random.binomial: lambda n, p: n * p * (1 - p),
    np.random.gamma: lambda shape, scale: shape * scale**2,
    np.random.exponential: lambda scale: scale**2,
}

def get_variance(dist, **dist_kwargs):
    return VARIANCE_FORMULAS[dist](**dist_kwargs)

###########################################################################################################################################################################


def svd_optimal_rankr(A, r):
# unique iff S[r-1]>S[r], otherwise any of the rth largest singular values chosen (depending on list S)

    """
    Computes an optimal solution for SVD rank-r truncation

    Parameters:
    A (nested list of depth 2): The matrix to be approximated
    r (int): The rank of the approximation

    Returns:
    A_r (numpy array): The rank-r approximation matrix, shape (m, n)
    U_r (numpy array): The first r left singular vectors of A, shape (m, r)
    S_r (nupy array): The diagonal matrix of the first r singular values, shape (r, n)
    Vt_r (numpy array): The first r right singular vectors of A, transposed, shape (r, n)
    """

    A=np.array(A)
    r = min(r, min(A.shape))
    
    U,S,Vt = np.linalg.svd(A, full_matrices=False)

    S_r = np.diag(S[:r])
    U_r = U[ : , :r]
    Vt_r = Vt[ :r, : ]

    A_r = U_r @ S_r @ Vt_r

    return A_r, U_r, S_r, Vt_r



def svd_error_rankr(A, r):

    """
    Computes Frobenius distance of rank-r approximation and the original matrix

    Parameters:
    A (nested list of depth 2): The original matrix 
    r (int): The rank of the approximation

    Returns:
    error (float): The error in the Frobenius norm
    """

    A_r, U_r, S_r, Vt_r = svd_optimal_rankr(A, r)
    error = np.linalg.norm(A_r - A, 'fro')

    return error



def random_matrix(shape, dist, **dist_kwargs):
    """
    Generates a matrix with entries drawn iid from 'dist' and with size 'shape'

    Parameters:
    shape (integer tuple): The size of the desired matrix, of the form (n, m)
    dist: The desired distribution - X - of entries, of the form np.random.X
    **dist_kwargs: The required parameters for the chosen distribution

    Returns:
    A (2D numpy array): The required matrix with entries drawn iid from 'dist' and with size 'shape'
    """
    A = dist(size=shape, **dist_kwargs)
    return A



# using normal distribution around optimal solution for perturbated matrices (any rank)
# scaled variance so that perturbations are reasonable, S_r[0,0] is largest singular value
def svd_free_rank_perturbation_plot(A, r, num_perturbed, **svd_type):

    """
    Generates scatter plot of perturbed matrices (with any rank) distance from A against distance from optimal rank-r approximation

    Parameters:
    A (nested list of depth 2): The original matrix 
    r (int): The rank of the approximation
    num_perturbed: The number of perturbated matrices to be plotted

    Returns:
    A scatter plot (shows how rank-r approx. is not necessarily best approx. amongst all ranks)
    """

    A_r, U_r, S_r, Vt_r = svd_optimal_rankr(A, r)

# list of perturbed matrices. vectored using 3D numpy array instead of loop
    A_perturbed = A_r + np.random.normal(0, 0.05*S_r[0,0], (num_perturbed, A_r.shape[0], A_r.shape[1]))

# x-axis: distance (Frobenius) of perturbed matrix from optimal rank-r truncation
    distance_approx = np.linalg.norm(A_perturbed - A_r, 'fro', axis=(1,2))
    distance_true = np.linalg.norm(A_perturbed - A, 'fro', axis=(1,2))
    
    plt.scatter(distance_approx, distance_true, color='blue', s=20)
    plt.scatter([0], svd_error_rankr(A, r), color='red')
    plt.xlabel(f"Frobenius dist. from rank-{r} optimal sol.")
    plt.ylabel("Frobenius dist. from true matrix")
    plt.title("Perturbation Scatter")
    plt.show()

###########################################################################################################################################################################

import pandas as pd


def error_ranksize_table(r, n, dist, **dist_kwargs):

    """
    Generates table showing error between original matrix of size (nxn) and optimal rank-r approximation 

    Parameters:
    r (int): The highest rank approximation desired
    n (int): The greatest number of rows for the matrix being approximated
    dist: The desired distribution - X - of entries, of the form np.random.X
    **dist_kwargs: The required parameters for the chosen distribution

    Returns:
    Table showing the error for each rank r and size nxn respectively 
    """
    rank_list = np.arange(1, r+1, 1)
    size_list = [f"{i}x{i}" for i in range(1, n+1, 1)]
    matrix_list = [dist(size=(i, i), **dist_kwargs) for i in range(1, n+1)]

    rankr_error_table = np.array([ [svd_error_rankr(matrix_list[j], rank_list[i]) for j in range(n)] for i in range(r)])

    table = pd.DataFrame(rankr_error_table, index=rank_list, columns=size_list)

    print(table)



def error_ranksize_plot(r, n, dist, **dist_kwargs):

    """
    Generates scatter showing error of the rank-r approx. against the size of the matrix (to see behaviour as n grows, whilst r stays the same)

    Parameters:
    r (int): The rank of approximation desired (fixed)
    n (int): The greatest number of rows for the matrix being approximated
    dist: The desired distribution - X - of entries, of the form np.random.X
    **dist_kwargs: The required parameters for the chosen distribution

    Returns:
    Scatter plot showing how error changes as size changes, for a fixed rank
    """

    matrix_list = [dist(size=(i, i), **dist_kwargs) for i in range(1, n+1)]
    rankr_error_list = np.array([svd_error_rankr(matrix_list[j], r) for j in range(n)])
    
    index_list = np.arange(1, n+1, 1)

    plt.scatter(index_list, rankr_error_list, color='red', s=20)
    plt.xlabel("#rows of the square matrix")
    plt.ylabel("Error in approx.")
    plt.title("Error against size (fixed rank)")
    plt.show()



def svd_singularvalues_gaussianmatrix(n, m, k):

    """
    Creates different plots showing information about singular values of Gaussian matrices

    Parameters:
    n (int): #rows of matrices
    m (int): #columns of matrices
    k (int): Total number of matrices to be used (sample size)

    Returns:
    Histogram, Boxplot to show dist. of singular values. Histogram of smallest and largest singular values to see behaviour at either end.
    """

    gaussian_list = np.random.normal(0, 1, (k, n, m))
    U,S,Vt = np.linalg.svd(gaussian_list, full_matrices=False)

    singularvalues_list = np.array(S).flatten()

# histogram of singular values:
    plt.hist(singularvalues_list, bins=100, density=True)
    plt.xlabel("Singular value")
    plt.ylabel("Density")
    plt.show()

# boxplot of singular values:  
    plt.boxplot(singularvalues_list)
    plt.xlabel("Singular value index")
    plt.ylabel("Value")
    plt.show()

# histogram of smallest and largest singular values
    largest_sv  = S[:, 0]
    smallest_sv = S[:, -1]

    S = np.array(S).flatten()

    plt.hist(largest_sv, bins=50, density=True, alpha=0.6, label="largest")
    plt.hist(smallest_sv, bins=50, density=True, alpha=0.6, label="smallest")
    plt.legend()
    plt.xlabel("Singular value")
    plt.ylabel("Density")
    plt.show()



# same rank perturbation
def svd_fixed_rank_perturbation_plot(A, r, num_perturbed, rotation=False):

    """
    Generates scatter plot of perturbed matrices (with fixed rank r) distance from A against distance from optimal rank-r approximation

    Parameters:
    A (nested list of depth 2): The original matrix 
    r (int): The rank of the approximation
    num_perturbed: The number of perturbated rank-r matrices to be plotted

    Returns:
    A scatter plot that shows optimality of A_r for that fixed rank r
    """

    A_r, U_r, S_r, Vt_r = svd_optimal_rankr(A, r)

    S_r = list(np.diag(S_r))

    try:
        k = S_r.index(0) 
    except ValueError:
        k = r

    S_r = np.array(S_r)

    pert_list = np.zeros((num_perturbed, r))
    pert_list[:, :k] = np.random.normal(0, 0.05*S_r[-1], (num_perturbed, k))

    sv_perturbed = S_r + pert_list
    S_r_pert_list = np.array([np.diag(i) for i in sv_perturbed])

    A_r_pert_list = U_r @ S_r_pert_list @ Vt_r

    if rotation: 

        Q_1, R_1 = np.linalg.qr(np.random.normal(0, 10**-5, (U_r.shape[0], Vt_r.shape[1])))
        if np.linalg.det(Q_1) < 0:
            Q_1[:, 0] = -Q_1[:, 0]

        Q_2, R_2 = np.linalg.qr(np.random.normal(0, 10**-5, (U_r.shape[0], Vt_r.shape[1])))
        if np.linalg.det(Q_2) < 0:
            Q_2[:, 0] = -Q_2[:, 0]
        
        A_r_pert_list = Q_1 @ A_r_pert_list @ Q_2

        distance_approx = np.linalg.norm(A_r_pert_list - A_r, 'fro', axis=(1,2))
        distance_true = np.linalg.norm(A_r_pert_list - A, 'fro', axis=(1,2))

        plt.scatter(distance_approx, distance_true, color='blue', s=20)
        plt.scatter([0], svd_error_rankr(A, r), color='red')
        plt.xlabel(f"Frobenius dist. from rank-{r} optimal sol.")
        plt.ylabel("Frobenius dist. from true matrix")
        plt.title("Rank-r Perturbation Scatter")
        plt.show()

    else:

        distance_approx = np.linalg.norm(A_r_pert_list - A_r, 'fro', axis=(1,2))
        distance_true = np.linalg.norm(A_r_pert_list - A, 'fro', axis=(1,2))

        plt.scatter(distance_approx, distance_true, color='blue', s=20)
        plt.scatter([0], svd_error_rankr(A, r), color='red')
        plt.xlabel(f"Frobenius dist. from rank-{r} optimal sol.")
        plt.ylabel("Frobenius dist. from true matrix")
        plt.title("Rank-r Perturbation Scatter")
        plt.show()


###########################################################################################################################################################################
from scipy.integrate import quad
from scipy.optimize import root_scalar

# Marchenko-Patur Law investigation

def mp_pdf(mu, lam, dist, **dist_kwargs):

    sigma2 = get_variance(dist, **dist_kwargs)

    mu_min = sigma2 * (1 - np.sqrt(lam))**2
    mu_max = sigma2 * (1 + np.sqrt(lam))**2

    if mu < mu_min or mu > mu_max:
        return 0

    numerator = np.sqrt((mu - mu_min) * (mu_max - mu))
    denominator = 2 * np.pi * sigma2 * lam * mu
    pdf = numerator / denominator

    return pdf



# To find mu_r numerically:
def find_mu_r(r, m, lam, dist, **dist_kwargs):

    sigma2 = get_variance(dist, **dist_kwargs)

    mu_min = sigma2 * (1 - np.sqrt(lam))**2
    mu_max = sigma2 * (1 + np.sqrt(lam))**2
    
    if r == 0:
        return mu_max  
    if r >= m:
        return mu_min

    def prob_above(upper_limit):
        return quad(lambda mu: mp_pdf(mu, lam, dist, **dist_kwargs), upper_limit, mu_max)[0]
    
    # Want prob_above(mu_r) = r/m
    target = r / m
    # Tolerance
    epsilon = 1e-6

    result = root_scalar(
        lambda mu: prob_above(mu) - target,
        bracket=[mu_min + epsilon, mu_max - epsilon],
        method='bisect'
    )
    
    return result.root



def expected_squared_error_theoretical(n, m, r, dist, **dist_kwargs):

    sigma2 = get_variance(dist, **dist_kwargs)
    lam = m / n
    
    if r == 0:
        return n * m * sigma2
    if r >= m:
        return 0.0
    
    mu_r = find_mu_r(r, m, lam, dist, **dist_kwargs)
    mu_min = sigma2 * (1 - np.sqrt(lam))**2
    
    integrand = lambda mu: mu * mp_pdf(mu, lam, dist, **dist_kwargs)
    integral, _ = quad(integrand, mu_min, mu_r)

    expected_error = (1 / lam) * integral

    return expected_error



def expected_squared_error_empirical_verification(n, m, r, k, dist, prnt=False, **dist_kwargs):

    matrix_list = dist(size=(k, n, m), **dist_kwargs)
    error_list = np.array([svd_error_rankr(A, r)**2 for A in matrix_list])

    mean_error = (1 / m) * np.mean(error_list)

    expected_error = expected_squared_error_theoretical(n, m, r, dist, **dist_kwargs)

    abs_error_diff = abs(expected_error - mean_error)

    if prnt:
    
        print(f'expected square error = {expected_error}')
        print(f'mean of squared error list = {mean_error}')
        print(f'absolute difference of errors = {abs_error_diff}')

    return mean_error

expected_squared_error_empirical_verification(5, 4, 0, 1000, np.random.normal, prnt=True, loc=0, scale=1)

def expected_squared_error_empirical_verification_plot(n, m, r, trials_x100, dist, **dist_kwargs):

    x_list = np.arange(100, 100*(trials_x100 + 1), 100)
    y_list = np.array([expected_squared_error_empirical_verification(n, m, r, i, dist, **dist_kwargs) for i in x_list])

    plt.scatter(x_list, y_list)
    plt.xlabel("Number of trials")
    plt.ylabel("Error")
    plt.title("Empirical squared error vs. number of trials")
    plt.show()


###########################################################################################################################################################################
   
if __name__ == "__main__":
# Testing on Gaussian matrices

    A = random_matrix((3,4), np.random.normal, loc=0, scale=1)

    svd_free_rank_perturbation_plot(A,1,1500) 
    svd_free_rank_perturbation_plot(A,2,1500) 
    svd_free_rank_perturbation_plot(A,3,1500) 

    error_ranksize_plot(10,100, np.random.normal, loc=0, scale=1)
    error_ranksize_plot(200,200, np.random.normal, loc=0, scale=1)

    svd_singularvalues_gaussianmatrix(3,3,10000)

    svd_fixed_rank_perturbation_plot(A,1,1500,) 
    svd_fixed_rank_perturbation_plot(A,2,1500,) 
    svd_fixed_rank_perturbation_plot(A,3,1500,) 

    svd_fixed_rank_perturbation_plot(A,1,1500, True) 
    svd_fixed_rank_perturbation_plot(A,2,1500, True) 
    svd_fixed_rank_perturbation_plot(A,3,1500, True) 
    svd_fixed_rank_perturbation_plot(A,1,1500, ) 
    svd_fixed_rank_perturbation_plot(A,2,1500, ) 
    svd_fixed_rank_perturbation_plot(A,3,1500, ) 

    print(f'rank 0 expected suared error is: {expected_squared_error_theoretical(5,4, 0, np.random.normal, loc=0, scale=1)}')
    print(f'rank 1 expected suared error is: {expected_squared_error_theoretical(5,4, 1, np.random.normal, loc=0, scale=1)}')
    print(f'rank 2 expected suared error is: {expected_squared_error_theoretical(5,4, 2, np.random.normal, loc=0, scale=1)}')
    print(f'rank 3 expected suared error is: {expected_squared_error_theoretical(5,4, 3, np.random.normal, loc=0, scale=1)}')
    print(f'rank 4 expected suared error is: {expected_squared_error_theoretical(5,4, 4, np.random.normal, loc=0, scale=1)}')

    expected_squared_error_empirical_verification(5, 4, 0, 1000, np.random.normal, prnt=True, loc=0, scale=1)
    expected_squared_error_empirical_verification(5, 4, 1, 1000, np.random.normal, prnt=True, loc=0, scale=1)
    expected_squared_error_empirical_verification(5, 4, 2, 1000, np.random.normal, prnt=True, loc=0, scale=1)
    expected_squared_error_empirical_verification(5, 4, 3, 1000, np.random.normal, prnt=True, loc=0, scale=1)
    expected_squared_error_empirical_verification(5, 4, 4, 1000, np.random.normal, prnt=True, loc=0, scale=1)

#   expected_squared_error_empirical_verification_plot(5, 4, 3, 500, np.random.normal, loc=0, scale=1)