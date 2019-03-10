import numpy as np
import parameters
import csv
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn import preprocessing


def GetRC(s):
    s = s.strip('r')
    l = s.split('_c')
    r = int(l[0]) - 1 #index starts at 0, while in csv it starts at 1
    c = int(l[1]) - 1
    return r,c


def WriteToCSV(data): # expect to be a matrix
    template = pd.read_csv(parameters.SAMPLECSV_PATH)
    print(template.values.shape)
    size = template.values.shape[0]
    rcstrs = [None] * size
    values = [0] * size
    count = 0
    for i in template.values:
        r, c = GetRC(i[0])
        rcstrs[count] = i[0]
        values[count] = data[r,c]
        count += 1
    df = pd.DataFrame({'Id': rcstrs,'Prediction': values})
    df.to_csv(parameters.OUTPUTCSV_PATH,index=False)


def preprocess(save = 0):
    rawdata = pd.read_csv(parameters.RAWDATA_PATH)
    datamat = np.zeros( (parameters.NROWS, parameters.NCOLS), dtype=np.float32 )
    total = 0
    count = 0
    for i in rawdata.values:
        r, c = GetRC(i[0])
        datamat[r,c] = i[1]
        total += i[1]
        count += 1
    fill = total/float(count)
    # if save:
    #     np.save(parameters.MATRAW_PATH, datamat)
    imputer = SimpleImputer(missing_values=0, strategy='constant', fill_value=fill)
    imputer = imputer.fit(datamat)
    newdata = imputer.transform(datamat)
    if save:
        np.save(parameters.MATMEAN_PATH, newdata)

    return newdata


def SVDBaseline(k=40, load = 0, save = 0):
    if load:
        data = np.load(parameters.MATMEAN_PATH)
    else:
        data = preprocess(save)

    if load:
        u = np.load('./cache/svdu.npy')
        s = np.load('./cache/svds.npy')
        vh = np.load('./cache/svdvh.npy')
    else:
        u, s, vh = np.linalg.svd(data)
        print (u.shape, s.shape, vh.shape)
        if save:
            np.save('./cache/svdu.npy', u)
            np.save('./cache/svds.npy', s)
            np.save('./cache/svdvh.npy', vh)
    # k = 0
    # totalsum = np.sum(np.square(s))
    # print(s)
    #
    # th = threshold * totalsum
    # current = 0
    # for i in s:
    #     k += 1;
    #     current += i*i
    #     if current > th:
    #         break;
    # print(k)
    mask = [1] * k + [0] * (1000 - k)
    sprime = mask*s
    recondata = np.dot(u[:, :1000] * sprime, vh)
    WriteToCSV(recondata)




if __name__ == "__main__":
    #preprocess()
    SVDBaseline(40, load = 1, save = 0)
    #data = np.load(parameters.MATMEAN_PATH)
