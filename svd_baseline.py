import numpy as np
import parameters
import csv
import pandas as pd
import imp
from util import *



def SVDBaseline(data, k=40):
    print("SVD Baseline: ")
    print("start svd")
    u, s, vh = np.linalg.svd(data)
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

def main():
    data = LoadHeuristicFill()
    SVDBaseline(data, k=40)



if __name__ == "__main__":
    main()
