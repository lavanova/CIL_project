import numpy as np
import parameters
from utils import *
from sklearn.impute import SimpleImputer

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

def MeanFill(data, mask):
    fill = float(np.sum(data))/np.sum(mask) # global mean
    imputer = SimpleImputer(missing_values=0, strategy='constant', fill_value=fill)
    imputer = imputer.fit(data)
    imputed_data = imputer.transform(data)
    return imputed_data

def ColFill(data, mask):
    col_mean = ( np.sum(data, axis=0) /np.sum(mask, axis=0) ).reshape(1, parameters.NCOLS) # 1 * 1000
    col_fill = (1 - mask) * (col_mean) + data
    return col_fill

def ColAdjFill(data, mask, K=25):
    global_mean = float(np.sum(data)) / np.sum(mask)
    col_mean = (np.sum(data, axis=0) / np.sum(mask, axis=0)).reshape(1, parameters.NCOLS)  # 1 * 1000
    col_sum = np.sum(mask, axis=0).reshape(1, parameters.NCOLS)  # 1 * 1000
    adj_col_mean = (col_mean * col_sum + global_mean * K) / (col_sum + K)
    col_fill = (1 - mask) * (adj_col_mean) + data
    return col_fill

def RowFill(data, mask):
    row_mean = (np.sum(data, axis=1) / np.sum(mask, axis=1)).reshape(parameters.NROWS, 1)  # 10000 * 1
    row_fill = (1 - mask) * row_mean + data
    return row_fill

def RowAdjFill(data, mask, K=25):
    global_mean = float(np.sum(data)) / np.sum(mask)
    row_mean = (np.sum(data, axis=1) / np.sum(mask, axis=1)).reshape(parameters.NROWS, 1)  # 10000 * 1
    row_sum = np.sum(mask, axis=1).reshape(parameters.NROWS, 1)  # 10000 * 1
    adj_row_mean = (row_mean * row_sum + global_mean * K) / (row_sum + K)
    adj_row_fill = (1 - mask) * adj_row_mean + data
    return adj_row_fill

def RateAdjustedFill(data, mask, K=20):
    global_mean = float(np.sum(data))/np.sum(mask)
    col_mean = ( np.sum(data, axis=0) /np.sum(mask, axis=0) ).reshape(1, parameters.NCOLS) # 1 * 1000
    col_sum = np.sum(mask, axis=0).reshape(1, parameters.NCOLS)    # 1 * 1000
    # print(np.min(col_sum))
    adj_col_mean = (col_mean * col_sum + global_mean * K) / (col_sum + K)
    row_mean = ( np.sum(data, axis=1)/np.sum(mask, axis=1) ).reshape(parameters.NROWS, 1) # 10000 * 1
    row_sum = np.sum(mask, axis=1).reshape(parameters.NROWS, 1)    # 10000 * 1
    # print(np.min(row_sum))
    adj_row_mean = (row_mean * row_sum + global_mean * K) / (row_sum + K)
    heur_fill = (1 - mask) * (adj_col_mean * 0.5 + adj_row_mean * 0.5) + data
    return heur_fill


def SVDBaseline(data, mask, k=50):
    # filled_data = MeanFill(data,mask)
    # filled_data = ColFill(data, mask)
    # filled_data = ColAdjFill(data, mask, K=20)
    # filled_data = RowFill(data, mask)
    # filled_data = RowAdjFill(data, mask, K=20)
    # filled_data = HeuristicFill(data, mask)
    filled_data = RateAdjustedFill(data, mask, K=10)

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
