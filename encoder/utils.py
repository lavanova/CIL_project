import numpy as np
import parameters
import csv
import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
import random

'''
str -> int,int
Transform RC strings to two integers eg. "r10_c12" -> 10,12
'''
def GetRC(s):
    s = s.strip('r')
    l = s.split('_c')
    r = int(l[0]) - 1 #index starts at 0, while in csv it starts at 1
    c = int(l[1]) - 1
    return r,c


'''
np.array (10000*1000), [path] -> None
Write the result file to the specific path,
default is: parameters.OUTPUTCSV_PATH  # './out.csv'
'''
def WriteToCSV(data, path = parameters.OUTPUTCSV_PATH): # expect the input to be a matrix
    
    print("writing result to csv")
    print(data.shape)
    template = pd.read_csv(parameters.SAMPLECSV_PATH)
    size = template.values.shape[0]
    rcstrs = [None] * size
    values = [0] * size
    count = 0
    for i in template.values:
        r, c = GetRC(i[0])
        rcstrs[count] = i[0]
        #print(r)
        #print(c)
        #print(data[r,c])
        values[count] = data[r,c]
        count += 1
    # data frame is reconstructed since the direct modification is too slow
    df = pd.DataFrame({'Id': rcstrs,'Prediction': values})
    df.to_csv(path,index=False)
    print("writing completed")


def LoadDataMask(save = 0, outpathdata = parameters.MATRAW_PATH,
    outpathmask = parameters.MASK_PATH, inpath = parameters.RAWDATA_PATH):
    rawdata = pd.read_csv(inpath)
    return getDataMask(rawdata.values)


def getDataMask(rawdata):
    data = np.zeros( (parameters.NROWS, parameters.NCOLS), dtype=np.float32 )
    mask = np.zeros( (parameters.NROWS, parameters.NCOLS), dtype=np.int )
    for i in rawdata:
        r, c = GetRC(i[0])
        data[r,c] = i[1]
        mask[r,c] = 1
    print("Load data mask complete")
    return data, mask


'''
split data into training and validation
valper: percentage of the validation data
'''
def LoadTrainValDataMask(valper=0.1, inpath = parameters.RAWDATA_PATH):
    rawdata = pd.read_csv(inpath)
    train_data = np.zeros( (parameters.NROWS, parameters.NCOLS), dtype=np.float32 )
    train_mask = np.zeros(  (parameters.NROWS, parameters.NCOLS), dtype=np.int )
    val_data = np.zeros( (parameters.NROWS, parameters.NCOLS), dtype=np.float32 )
    val_mask = np.zeros( (parameters.NROWS, parameters.NCOLS), dtype=np.int )
    for i in rawdata.values:
        r, c = GetRC(i[0])
        dice = random.uniform(0, 1)
        if dice < valper:
            val_data[r,c] = i[1]
            val_mask[r,c] = 1
        else:
            train_data[r,c] = i[1]
            train_mask[r,c] = 1
    print("Load train validation data mask complete")
    return train_data, train_mask, val_data, val_mask

'''
[int, outpath, inpath] -> np.array (10000*1000)
Load data from the file position, default: parameters.RAWDATA_PATH # './data/data_train.csv'
Impute the missing values to global mean
If save = 1
Save data to the outpath, default: parameters.MATMEAN_PATH # MATMEAN_PATH = './cache/matimpute.npy'
'''
def LoadMeanImpute(save = 0, outpath = parameters.MATMEAN_PATH, inpath = parameters.RAWDATA_PATH):
    data, mask = LoadDataMask()
    fill = float(np.sum(data))/np.sum(mask) # global mean
    imputer = SimpleImputer(missing_values=0, strategy='constant', fill_value=fill)
    imputer = imputer.fit(data)
    imputed_data = imputer.transform(data)
    if save:
        np.save(outpath, imputed_data)
    print("Load mean imputeing complete")
    return imputed_data

'''
Dense implementation of validation cost
The loss is MSE
'''
def getMSE(pred, data, mask):
    assert(data.shape == pred.shape), "The input matrix dimensions mismatch"
    return np.sum( (mask * (data - pred)) ** 2 ) / np.sum(mask)

'''
Get RMSE error of the validation data
'''
def getRMSE(pred, data, mask):
    mse = getMSE(pred, data, mask)
    return np.sqrt(mse)

'''
[int, outpath, inpath] -> np.array (10000*1000) in Compressed row array format
'''
def LoadCSR(save = 0, outpath = parameters.MATCSR_PATH, inpath = parameters.RAWDATA_PATH):
    rawdata = pd.read_csv(inpath)
    row = []
    col = []
    data = []
    for i in rawdata.values:
        r, c = GetRC(i[0])
        row.append(r)
        col.append(c)
        data.append(i[1])
    newdata = csr_matrix((data, (row, col)), shape=(parameters.NROWS, parameters.NCOLS))
    #print(newdata.toarray().shape)
    # print( np.mean(data) )
    # print( np.std(data) )
    data = np.array(data)
    return newdata

if __name__ == "__main__":
    #data = LoadMeanImpute()
    data = LoadHeuristicFill()
    print(data.shape)
    print(data)

    # pred = np.array([[1,2],[3,4]])
    # valdata = np.array([[1,1],[1,1]])
    # valmask = np.array([[0,1],[0,1]])
    # print(valLoss(pred, valdata, valmask))
