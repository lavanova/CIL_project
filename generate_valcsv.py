import pandas as pd
from sklearn.model_selection import KFold
from utils import *
from ALS import ALS
from svd_baseline import *
import numpy as np


'''
Main function for getting validation for ALS and SVD
'''


# filled_data = MeanFill(data,mask)
# filled_data = ColFill(data, mask)
# filled_data = ColAdjFill(data, mask, K=10)
# filled_data = RowFill(data, mask)
# filled_data = RowAdjFill(data, mask, K=10)
# filled_data = HeuristicFill(data, mask)
# filled_data = RateAdjustedFill(data, mask, K=10)


def dense_model(model=SVDBaseline, fcn = RateAdjustedFill, opath = 'svd.csv'):
    train_data, train_mask, val_data, val_mask = LoadFixedValDataMask()
    result_data = model(train_data, train_mask, fcn)
    rmse = getRMSE(result_data, val_data, val_mask)
    print("Validation loss is:", rmse)
    WriteToCSV(result_data, path=opath, sample=parameters.VALTRUTH_PATH)


def svd_main():
    loadfcns = [MeanFill, ColFill, ColAdjFill, RowFill, RowAdjFill, HeuristicFill, RateAdjustedFill]
    model_names = ['svd_mean','svd_col','svd_coladj','svd_row','svd_rowadj','svd_heuristic', 'svd_rateadjust']
    opaths = [None for i in range(len(model_names))]
    prefix = 'cache/'
    suffix = ''
    for i in range(len(model_names)):
        opath = prefix + model_names[i] + suffix
        opaths[i] = opath
        dense_model(model=SVDBaseline, fcn=loadfcns[i], opath=opath)



if __name__ == "__main__":
    # dense_model(model = SVDBaseline, opath = 'svd_val.csv')
    svd_main()