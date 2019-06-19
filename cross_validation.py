import pandas as pd
from sklearn.model_selection import KFold
from utils import *
from ALS import ALS


def cross_validate(nfold=5, inpath=parameters.RAWDATA_PATH):
    rawdata = pd.read_csv(inpath)
    X = rawdata.values
    kf = KFold(n_splits=nfold, shuffle=True)
    kf.get_n_splits(X)
    results = []
    for train_index, test_index in kf.split(X):
        print("Starting {}th fold ".format(len(results)+1))
        X_train, X_test = X[train_index], X[test_index]
        train_data, train_mask = getDataMask(X_train)
        result_data = ALS(train_data, train_mask, epochs=4, factors=8, regularizer=0.05)
        test_data, test_mask = getDataMask(X_test)
        rmse = getRMSE(result_data,test_data,test_mask)
        print("Validation RMSE is: {}".format(rmse))
        results.append(rmse)
    print(results)
    avgrmse = np.mean(results)
    print("The average rmse is: {}".format(avgrmse))
    return avgrmse


if __name__ == "__main__":
    cross_validate(nfold=5)