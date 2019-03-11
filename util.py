import numpy as np
import parameters
import csv
import pandas as pd
from sklearn.impute import SimpleImputer

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
    template = pd.read_csv(parameters.SAMPLECSV_PATH)
    size = template.values.shape[0]
    rcstrs = [None] * size
    values = [0] * size
    count = 0
    for i in template.values:
        r, c = GetRC(i[0])
        rcstrs[count] = i[0]
        values[count] = data[r,c]
        count += 1
    # data frame is reconstructed since the direct modification is too slow
    df = pd.DataFrame({'Id': rcstrs,'Prediction': values})
    df.to_csv(path,index=False)
    print("writing completed")

'''
[int, outpath, inpath] -> np.array (10000*1000)
Load data from the file position, default: parameters.RAWDATA_PATH # './data/data_train.csv'
Impute the missing values to global mean
If save = 1
Save data to the outpath, default: parameters.MATMEAN_PATH # MATMEAN_PATH = './cache/matimpute.npy'
'''
def LoadMeanImpute(save = 0, outpath = parameters.MATMEAN_PATH, inpath = parameters.RAWDATA_PATH):
    rawdata = pd.read_csv(inpath)
    datamat = np.zeros( (parameters.NROWS, parameters.NCOLS), dtype=np.float32 )
    total = 0
    count = 0
    for i in rawdata.values:
        r, c = GetRC(i[0])
        datamat[r,c] = i[1]
        total += i[1]
        count += 1
    fill = total/float(count) # global mean
    imputer = SimpleImputer(missing_values=0, strategy='constant', fill_value=fill)
    imputer = imputer.fit(datamat)
    newdata = imputer.transform(datamat)
    if save:
        np.save(outpath, newdata)
    print("Load mean imputeing complete")
    return newdata
