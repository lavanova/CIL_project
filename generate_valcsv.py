import pandas as pd
from sklearn.model_selection import KFold
from utils import *
from ALS import ALS
from svd_baseline import SVDBaseline
import numpy as np


'''
Main function for getting validation for ALS and SVD
'''

def dense_main(model=SVDBaseline, opath = 'svd.csv'):
    train_data, train_mask, val_data, val_mask = LoadFixedValDataMask()
    result_data = model(train_data, train_mask)
    rmse = getRMSE(result_data, val_data, val_mask)
    print("Validation loss is:", rmse)
    WriteToCSV(result_data, path=opath, sample=parameters.SAMPLECSV_PATH)


if __name__ == "__main__":
    dense_main(model = SVDBaseline, opath = 'svd_val.csv')