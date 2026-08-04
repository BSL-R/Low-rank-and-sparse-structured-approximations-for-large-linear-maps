import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root_scalar

import svd_investigation as svd

# Marchenko-Patur Law investigation (particular to Frobenius norm and svd_optimal)

def mp_pdf(mu, lam, dist, **dist_kwargs):

    sigma2 = svd.get_variance(dist, **dist_kwargs)

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

    sigma2 = svd.get_variance(dist, **dist_kwargs)

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

    sigma2 = svd.get_variance(dist, **dist_kwargs)
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
    error_list = np.array([svd.svd_error(A, r, 'fro', svd.svd_optimal, {})**2 for A in matrix_list])

    mean_error = (1 / m) * np.mean(error_list)

    expected_error = expected_squared_error_theoretical(n, m, r, dist, **dist_kwargs)

    abs_error_diff = abs(expected_error - mean_error)

    if prnt:
    
        print(f'expected square error = {expected_error}')
        print(f'mean of squared error list = {mean_error}')
        print(f'absolute difference of errors = {abs_error_diff}')

    return mean_error



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

    expected_squared_error_empirical_verification_plot(5, 4, 3, 500, np.random.normal, loc=0, scale=1)