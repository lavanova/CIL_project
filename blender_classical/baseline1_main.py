import pandas as pd
from sklearn.model_selection import KFold
from utils import *
from ALS import ALS
from svd_baseline import *
import numpy as np
from surprise_models import surprise_model

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


def dense_model(model=SVDBaseline, fcn = RateAdjustedFill, valopath = '../cache/default', testopath = '../test/default'):
    train_data, train_mask, val_data, val_mask = LoadFixedValDataMask()
    result_data = model(train_data, train_mask, fcn)
    rmse = getRMSE(result_data, val_data, val_mask)
    print("Validation loss is:", rmse)
    WriteToCSV(result_data, path=valopath, sample=parameters.VALTRUTH_PATH)
    WriteToCSV(result_data, path=testopath, sample=parameters.SAMPLECSV_PATH)


def ALSmodel(epochs=4, factors=10, regularizer=1, valopath = '../cache/default', testopath = '../test/default'):
    train_data, train_mask, val_data, val_mask = LoadFixedValDataMask()
    result_data = ALS(train_data, train_mask, epochs=epochs, factors=factors, regularizer=regularizer)
    rmse = getRMSE(result_data, val_data, val_mask)
    print("Validation loss is:", rmse)
    WriteToCSV(result_data, path=valopath, sample=parameters.VALTRUTH_PATH)
    WriteToCSV(result_data, path=testopath, sample=parameters.SAMPLECSV_PATH)


def svd_main(model_names = ['svd_mean','svd_col','svd_coladj','svd_row','svd_rowadj','svd_heuristic', 'svd_rateadjust']):
    loadfcns = [MeanFill, ColFill, ColAdjFill, RowFill, RowAdjFill, HeuristicFill, RateAdjustedFill]
    valprefix = '../cache/'
    testprefix = '../test/'
    for i in range(len(model_names)):
        valopath = valprefix + model_names[i]
        testopath = testprefix + model_names[i]
        dense_model(model=SVDBaseline, fcn=loadfcns[i], valopath=valopath, testopath=testopath)


def surprise_main(item=True, user=True, slope=True, svdp=True):
    surprise_model(item=item, user=user, slope=slope, svdp=svdp)


def ALS_main(small=True, medium=True, big=True):
    valprefix = '../cache/'
    testprefix = '../test/'
    if small:
        name = 'ALS_small'
        ALSmodel(epochs=6, factors=5, regularizer=15, valopath=valprefix + name, testopath=testprefix + name)
    if medium:
        name = 'ALS_medium'
        ALSmodel(epochs=5, factors=20, regularizer=20, valopath=valprefix + name, testopath=testprefix + name)
    if big:
        name = 'ALS_big'
        ALSmodel(epochs=5, factors=50, regularizer=25, valopath=valprefix + name, testopath=testprefix + name)


if __name__ == "__main__":
    svd_main(model_names = ['svd_mean','svd_col','svd_coladj','svd_row','svd_rowadj','svd_heuristic', 'svd_rateadjust'])
    ALS_main(small=True, medium=True, big=True)
    surprise_main(item=True, user=True, slope=True, svdp=True)