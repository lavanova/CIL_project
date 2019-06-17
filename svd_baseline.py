import numpy as np
import parameters
from utils import *

'''
Load and compute heuristic average
'''
def HeuristicFill(data, mask):
    global_mean = float(np.sum(data))/np.sum(mask)
    col_mean = ( np.sum(data, axis=0) /np.sum(mask, axis=0) ).reshape(1, parameters.NCOLS) # 1 * 1000
    col_sum = np.sum(mask, axis=0).reshape(1, parameters.NCOLS)    # 1000 * 1
    row_mean = ( np.sum(data, axis=1)/np.sum(mask, axis=1) ).reshape(parameters.NROWS, 1) # 10000 * 1
    row_sum = np.sum(mask, axis=1).reshape(parameters.NROWS, 1)    # 1000 * 1
    heur_fill = (1 - mask) * (col_mean * 0.5 + row_mean * 0.5) + data
    return heur_fill

def SVDBaseline(data, mask, k=40):
    filled_data = HeuristicFill(data, mask)
    print("SVD Baseline: ")
    print("start svd")
    u, s, vh = np.linalg.svd(filled_data)
    print("finish svd")
    # k = 0
    # totalsum = np.sum(np.square(s))
    # print(s)
    # th = threshold * totalsum
    # current = 0
    # for i in s:
    #     k += 1;
    #     current += i*i
    #     if current > th:
    #         break;
    # print(k)
    diag_mask = [1] * k + [0] * (1000 - k)
    sprime = diag_mask*s
    recondata = np.dot(u[:, :1000] * sprime, vh)
    return recondata

def main(path='svd_baseline.csv'):
    data, mask = LoadDataMask()
    pred = SVDBaseline(data, mask, k=40)
    print(getRMSE(pred, data, mask))
    WriteToCSV(pred, path=path)

if __name__ == "__main__":
    main(path='svd_baseline.csv')
