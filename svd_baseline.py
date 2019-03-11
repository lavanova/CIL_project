import numpy as np
import parameters
import csv
import pandas as pd
import imp
from util import *



def SVDBaseline(k=40, load = 0, save = 0):
    if load:
        data = np.load(parameters.MATMEAN_PATH)
    else:
        data = LoadMeanImpute(save)

    if load:
        u = np.load('./cache/svdu.npy')
        s = np.load('./cache/svds.npy')
        vh = np.load('./cache/svdvh.npy')
    else:
        print("start svd")
        u, s, vh = np.linalg.svd(data)
        #print (u.shape, s.shape, vh.shape)
        if save:
            np.save('./cache/svdu.npy', u)
            np.save('./cache/svds.npy', s)
            np.save('./cache/svdvh.npy', vh)
        print("finish svd")
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
    SVDBaseline(40, load = 0, save = 0)
