import implicit
import scipy
import numpy as np
from scipy.sparse import csr_matrix
import imp
from util import *


def AMS(data, factors=60, regularization=0.01):
    model = implicit.als.AlternatingLeastSquares(factors=factors, regularization = regularization)
    model.fit(data)
    # recommend items for a user
    user_items = data.T.tocsr()
    print(model.item_factors.shape) #10000*60
    print(model.user_factors.shape) #1000*60
    # recommendations = model.recommend(1, user_items)
    # print(recommendations)
    recondata = np.matmul(model.item_factors, np.transpose(model.user_factors))
    meanrecon = np.mean(recondata)
    stdrecon = np.std(recondata)
    meanori = 3.86
    stdori = 1.12
    sigma = stdori/stdrecon
    recondata = (recondata - meanrecon) * sigma + meanori #rescale data to 1-5 scale
    np.clip(recondata, 1, 5, out=recondata)

    print(np.mean(recondata))
    print(np.std(recondata))
    WriteToCSV(recondata)


def main():
    data = LoadCSR()
    AMS(data, factors=60, regularization=0.01)

if __name__ == "__main__":
    main()
