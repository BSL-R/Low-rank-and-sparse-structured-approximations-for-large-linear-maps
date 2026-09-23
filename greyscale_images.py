import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import torch
from torchvision import datasets, transforms

import svd_investigation as svd


###########################################################################################################################################################################
   
# MNIST dataset (application)
mnist = datasets.MNIST( root="./data", train=True, download=True, transform=transforms.ToTensor())

def entry_mnist(i):

    image, label = mnist[i]
    A = image.squeeze().numpy()

    return A, label



# CIFAR-10 dataset (application)
cifar = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]))

def entry_cifar(i):

    image, label = cifar[i]
    A = image.squeeze().numpy()

    return A, label


###########################################################################################################################################################################


def image_comparison(A, r, svd_type, svd_type_kwargs):

    """
    Produces side-by-side image of original and approx.

    Parameters:
    A (matrix): The matrix representative of the image
    r (int): The rank of the approximation
    svd_type: SVD approx. method to be used. eg: svd_optimal, svd_random
    svd_type_kwargs:
    """

    A_r, U_r, S_r, Vt_r = svd_type(A, r, **svd_type_kwargs)
    

    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.imshow(A, cmap="gray")
    plt.title(f"Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(A_r, cmap="gray")
    plt.title(f"Rank-{r} approx.")
    plt.axis("off")

    plt.show()



def error_runtime_comparison(A, r, num_iter, norm_type, svd_type1, svd_type1_kwargs, svd_type2, svd_type2_kwargs):

    error1_list = np.empty(num_iter)
    error2_list = np.empty(num_iter)
    runtime1_list = np.empty(num_iter)
    runtime2_list = np.empty(num_iter)

    for k in range(num_iter):
        
        start = time.perf_counter()
        result1 = svd_type1(A, r, **svd_type1_kwargs)
        end = time.perf_counter()
        runtime1_list[k] = end - start

        if result1 is None:
            return np.nan, np.nan, np.nan, np.nan
        
        A_r1, U1, S1, Vt1 = result1

        
        start = time.perf_counter()
        result2 = svd_type2(A, r, **svd_type2_kwargs)
        end = time.perf_counter()
        runtime2_list[k] = end - start

        if result2 is None:
            return np.nan, np.nan, np.nan, np.nan
        
        A_r2, U2, S2, Vt2 = result2


        error1_list[k] = np.linalg.norm(A_r1 - A, ord=norm_type)
        error2_list[k] = np.linalg.norm(A_r2 - A, ord=norm_type)

    error1 = error1_list.mean()
    error2 = error2_list.mean()
    runtime1 = runtime1_list.mean()
    runtime2 = runtime2_list.mean()

    return error1, error2, runtime1, runtime2



def error_runtime_comparison_plot(A, r_max, num_iter, norm_type, svd_type1, svd_type1_kwargs, svd_type2, svd_type2_kwargs):

    error1_list = np.empty(r_max)
    error2_list = np.empty(r_max)
    runtime1_list = np.empty(r_max)
    runtime2_list = np.empty(r_max)

    for k in range(r_max):
        
        error1, error2, runtime1, runtime2 = error_runtime_comparison(A, k, num_iter, norm_type, svd_type1, svd_type1_kwargs, svd_type2, svd_type2_kwargs)

        error1_list[k] = error1
        error2_list[k] = error2
        runtime1_list[k] = runtime1
        runtime2_list[k] = runtime2

    plt.plot(runtime1_list, error1_list, 'o-', label="svd_optimal")
    plt.plot(runtime2_list, error2_list, 's-', label="svd_random")
    plt.title(f"error against runtime for each rank up to {r_max}")
    plt.xlabel("Runtime (s)")
    plt.ylabel("Error")
    plt.legend()
    plt.show()



def tiled_matrix(rows, cols, entry_type, start):

    images = []
    k = start

    for i in range(rows):
        row = []
        for j in range(cols):
            A, label = entry_type(k)
            row.append(A)
            k = k + 1
        images.append(np.hstack(row))

    A = np.vstack(images)

    return A


A = tiled_matrix(3, 3, entry_mnist, 2) #84x84 matrix
error_runtime_comparison_plot(A, 84, 100, 'fro', svd.svd_optimal, {}, svd.svd_random, {'q':0})
###########################################################################################################################################################################
   
if __name__ == "__main__":

    A, label = entry_mnist(12345)
    image_comparison(A, 10, svd.svd_optimal, {})
    image_comparison(A, 10, svd.svd_random, {'q':0})
    image_comparison(A, 10, svd.svd_random, {'q':1})
    image_comparison(A, 10, svd.svd_random, {'q':2})

    distance1, distance2, runtime1, runtime2 = error_runtime_comparison(A, 10, 1000, 'fro', svd.svd_optimal, {}, svd.svd_random, {'q':0})
    print(f'Avg. distance from original matrix: method1 = {distance1}, method2 = {distance2}\nAvg. runtime diff. (method1 - method2) = {runtime1 - runtime2} seconds')
    distance1, distance2, runtime1, runtime2 = error_runtime_comparison(A, 10, 1000, 'fro', svd.svd_optimal, {}, svd.svd_random, {'q':1})
    print(f'Avg. distance from original matrix: method1 = {distance1}, method2 = {distance2}\nAvg. runtime diff. (method1 - method2) = {runtime1 - runtime2} seconds')
    distance1, distance2, runtime1, runtime2 = error_runtime_comparison(A, 10, 1000, 'fro', svd.svd_optimal, {}, svd.svd_random, {'q':2})
    print(f'Avg. distance from original matrix: method1 = {distance1}, method2 = {distance2}\nAvg. runtime diff. (method1 - method2) = {runtime1 - runtime2} seconds')

    error_runtime_comparison_plot(A, 15, 1000, 'fro', svd.svd_optimal, {}, svd.svd_random, {'q':0})
    A = tiled_matrix(3, 3, entry_mnist, 2) #84x84 matrix
    error_runtime_comparison_plot(A, 84, 100, 'fro', svd.svd_optimal, {}, svd.svd_random, {'q':0})