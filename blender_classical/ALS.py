'''
ALS.py:Alternating least square implementation using numpy
'''

import scipy
import numpy as np
from scipy.sparse import csr_matrix
from utils import *
from tqdm import trange


'''
In place update U position
dense matrix implementation
'''
def ALS_iterU(data, mask, U, V, regularizer=0.01):
    n = U.shape[1]
    m = V.shape[1]
    k = V.shape[0]
    for i in range(n):
        Vm = ( mask[i,:]*V )
        mat = np.linalg.inv(  Vm @ Vm.T + regularizer*np.identity(k) ) # k*k
        val = V @ (mask[i, :]*data[i, :]).T  # k*1
        U[:,i] = mat @ val
    return U


def ALS_iterV(data, mask, U, V, regularizer=0.01):
    return ALS_iterU(data.T, mask.T, V, U, regularizer)


def calc_cost(data, mask, U, V):
    # print((U.T @ V))
    return getRMSE((U.T @ V), data, mask)

'''
Dense implementation of ALS
'''
def ALS(data, mask, epochs = 5, factors=50, regularizer=0.01):
    n = parameters.NROWS
    m = parameters.NCOLS
    k = factors
    U = np.random.normal(1, 1, (k, n))
    V = np.random.normal(1, 1, (k, m))

    for i in range(epochs):
        print("Iteration: " + str(i))
        ALS_iterU(data, mask, U, V, regularizer=regularizer)
        print(calc_cost(data, mask, U, V))
        ALS_iterV(data, mask, U, V, regularizer=regularizer)
        print(calc_cost(data, mask, U, V))

    recondata = np.clip(U.T @ V, a_min = 1, a_max = 5)
    return recondata


def main(epochs=5, factors=40, regularizer=0.01, path='ALS.csv'):
    load = 0
    if load:
        data = np.load(parameters.MATRAW_PATH)
        mask = np.load(parameters.MASK_PATH)
    else:
        data, mask = LoadDataMask()
    pred = ALS(data, mask, epochs=epochs, factors=factors, regularizer=regularizer)
    WriteToCSV(pred, path=path)

if __name__ == "__main__":
    main(epochs=5, factors=40, regularizer=0.1, path='ALS.csv')
