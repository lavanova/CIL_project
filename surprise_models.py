from surprise import *
import numpy as np
import pandas as pd
import parameters

def submission_table(original_df, col_userID, col_movie, col_rate):
    """ return table according with Kaggle convention """

    def id(row):
        return 'r' + str(int(row[col_userID])) + '_c' + str(int(row[col_movie]))

    def pred(row):
        return row[col_rate]

    df = pd.DataFrame.copy(original_df)
    df['Id'] = df.apply(id, axis=1)
    df['Prediction'] = df.apply(pred, axis=1)

    return df[['Id', 'Prediction']]

def load_csv(filename='data/data_train.csv'):
    """
    Function to load as a pandas dataframe a csv dataset in the standard format

    Args:
        filename (str): the csv file to read. It should be a table with columns Id, Prediction,
            with Id in the form r44_c1 where 44 is the user and 1 is the item

    Returns:
        pandas.DataFrame: ['User', 'Movie', 'Rating']
    """

    df = pd.read_csv(filename)
    df['User'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
    df['Movie'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
    df['Rating'] = df['Prediction']
    df = df.drop(['Id', 'Prediction'], axis=1)
    return df

def knn(train, validation, test, fn, **kwargs):
    """
    K Nearest Neighbors with Baseline from library Surprise
    """
    # Get parameters
    k = kwargs['k']
    sim_options = kwargs['sim_options']

    # First, we need to dump the pandas DF into files
    train_file = 'tmp_train.csv'
    test_file = 'tmp_test.csv'
    val_file = 'tmp_val.csv'
    train.to_csv(train_file, index=False, header=False)
    test.to_csv(test_file, index=False, header=False)
    validation.to_csv(val_file, index=False, header=False)

    # Create Reader
    reader = Reader(line_format='user item rating', sep=',')

    # Train and test set for Surprise
    fold_val = [(train_file, val_file)]
    fold_test = [(train_file, test_file)]

    # Load the data
    data_val = Dataset.load_from_folds(fold_val, reader=reader)
    data_test = Dataset.load_from_folds(fold_test, reader=reader)

    # Algorithm
    algo = KNNBaseline(k=k, sim_options=sim_options)

    for trainset, valset in data_val.folds():
        # Train
        algo.train(trainset)
        # Validation
        valpred = algo.test(valset)

    for trainset, testset in data_test.folds():
        testpred = algo.test(testset)

    # Clipping:
    tpred = np.zeros(len(testpred))
    vpred = np.zeros(len(valpred))
    for i in range(len(testpred)):
        val = testpred[i].est
        if val > 5:
            tpred[i] = 5
        elif val < 1:
            tpred[i] = 1
        else:
            tpred[i] = val

    for i in range(len(valpred)):
        val = valpred[i].est
        if val > 5:
            vpred[i] = 5
        elif val < 1:
            vpred[i] = 1
        else:
            vpred[i] = val

    # Copy the test
    test_df = test.copy()
    test_df.Rating = tpred

    val_df = validation.copy()
    val_df.Rating = vpred


    val_submission = submission_table(val_df, 'User', 'Movie', 'Rating')
    test_submission = submission_table(test_df, 'User', 'Movie', 'Rating')

    val_fn = 'cache/' + fn
    test_fn = 'test/' + fn

    print("Writing test file:")
    test_submission.to_csv(test_fn, index=False)
    print("Writing validation file:")
    val_submission.to_csv(val_fn, index=False)


def SVDsuprise(train, validation, test, fn, **kwargs):
    """
    SVD++ from library Surprise
    """
    # Get parameters
    n_factors = kwargs['n_factors']
    n_epochs = kwargs['n_epochs']
    reg_all = kwargs['reg_all']
    lr_all = kwargs['lr_all']

    # First, we need to dump the pandas DF into files
    train_file = 'tmp_train.csv'
    test_file = 'tmp_test.csv'
    val_file = 'tmp_val.csv'
    train.to_csv(train_file, index=False, header=False)
    test.to_csv(test_file, index=False, header=False)
    validation.to_csv(val_file, index=False, header=False)

    # Create Reader
    reader = Reader(line_format='user item rating', sep=',')

    # Train and test set for Surprise
    fold_val = [(train_file, val_file)]
    fold_test = [(train_file, test_file)]

    # Load the data
    data_val = Dataset.load_from_folds(fold_val, reader=reader)
    data_test = Dataset.load_from_folds(fold_test, reader=reader)

    # Algorithm
    # algo = SVDpp(sim_options=sim_options)
    algo = SVDpp(n_factors=n_factors, n_epochs=n_epochs, reg_all=reg_all, lr_all=lr_all, verbose=True)

    for trainset, valset in data_val.folds():
        # Train
        algo.fit(trainset)
        # Validation
        valpred = algo.test(valset)

    for trainset, testset in data_test.folds():
        testpred = algo.test(testset)

    # Clipping:
    tpred = np.zeros(len(testpred))
    vpred = np.zeros(len(valpred))
    for i in range(len(testpred)):
        val = testpred[i].est
        if val > 5:
            tpred[i] = 5
        elif val < 1:
            tpred[i] = 1
        else:
            tpred[i] = val

    for i in range(len(valpred)):
        val = valpred[i].est
        if val > 5:
            vpred[i] = 5
        elif val < 1:
            vpred[i] = 1
        else:
            vpred[i] = val

    # Copy the test
    test_df = test.copy()
    test_df.Rating = tpred

    val_df = validation.copy()
    val_df.Rating = vpred


    val_submission = submission_table(val_df, 'User', 'Movie', 'Rating')
    test_submission = submission_table(test_df, 'User', 'Movie', 'Rating')

    val_fn = 'cache/' + fn
    test_fn = 'test/' + fn

    print("Writing test file:")
    test_submission.to_csv(test_fn, index=False)
    print("Writing validation file:")
    val_submission.to_csv(val_fn, index=False)

def NMFsuprise(train, validation, test, fn, **kwargs):
    """
    SVD++ from library Surprise
    """
    # Get parameters
    n_factors = kwargs['n_factors']
    n_epochs = kwargs['n_epochs']
    biased = kwargs['biased']
    reg_pu = kwargs['reg_pu']
    reg_qi = kwargs['reg_qi']
    reg_bu = kwargs['reg_bu']
    reg_bi = kwargs['reg_bi']
    lr_bu = kwargs['lr_bu']
    lr_bi = kwargs['lr_bi']
    init_low = kwargs['init_low']
    init_high = kwargs['init_high']



    # First, we need to dump the pandas DF into files
    train_file = 'tmp_train.csv'
    test_file = 'tmp_test.csv'
    val_file = 'tmp_val.csv'
    train.to_csv(train_file, index=False, header=False)
    test.to_csv(test_file, index=False, header=False)
    validation.to_csv(val_file, index=False, header=False)

    # Create Reader
    reader = Reader(line_format='user item rating', sep=',')

    # Train and test set for Surprise
    fold_val = [(train_file, val_file)]
    fold_test = [(train_file, test_file)]

    # Load the data
    data_val = Dataset.load_from_folds(fold_val, reader=reader)
    data_test = Dataset.load_from_folds(fold_test, reader=reader)

    # Algorithm
    # algo = NMF
    algo = NMF(n_factors=n_factors, n_epochs=n_epochs, biased=biased, reg_pu=reg_pu, reg_qi=reg_qi,
               reg_bu=reg_bu, reg_bi=reg_bi, lr_bu=lr_bu, lr_bi=lr_bi, init_low=init_low,
               init_high=init_high, verbose=True)

    for trainset, valset in data_val.folds():
        # Train
        algo.fit(trainset)
        # Validation
        valpred = algo.test(valset)

    for trainset, testset in data_test.folds():
        testpred = algo.test(testset)

    # Clipping:
    tpred = np.zeros(len(testpred))
    vpred = np.zeros(len(valpred))
    for i in range(len(testpred)):
        val = testpred[i].est
        if val > 5:
            tpred[i] = 5
        elif val < 1:
            tpred[i] = 1
        else:
            tpred[i] = val

    for i in range(len(valpred)):
        val = valpred[i].est
        if val > 5:
            vpred[i] = 5
        elif val < 1:
            vpred[i] = 1
        else:
            vpred[i] = val

    # Copy the test
    test_df = test.copy()
    test_df.Rating = tpred

    val_df = validation.copy()
    val_df.Rating = vpred


    val_submission = submission_table(val_df, 'User', 'Movie', 'Rating')
    test_submission = submission_table(test_df, 'User', 'Movie', 'Rating')

    val_fn = 'cache/' + fn
    test_fn = 'test/' + fn

    print("Writing test file:")
    test_submission.to_csv(test_fn, index=False)
    print("Writing validation file:")
    val_submission.to_csv(val_fn, index=False)


def CoClusteringsuprise(train, validation, test, fn, **kwargs):
    """
    SVD++ from library Surprise
    """
    # Get parameters
    n_cltr_u = kwargs['n_cltr_u']
    n_cltr_i = kwargs['n_cltr_i']
    n_epochs = kwargs['n_epochs']



    # First, we need to dump the pandas DF into files
    train_file = 'tmp_train.csv'
    test_file = 'tmp_test.csv'
    val_file = 'tmp_val.csv'
    train.to_csv(train_file, index=False, header=False)
    test.to_csv(test_file, index=False, header=False)
    validation.to_csv(val_file, index=False, header=False)

    # Create Reader
    reader = Reader(line_format='user item rating', sep=',')

    # Train and test set for Surprise
    fold_val = [(train_file, val_file)]
    fold_test = [(train_file, test_file)]

    # Load the data
    data_val = Dataset.load_from_folds(fold_val, reader=reader)
    data_test = Dataset.load_from_folds(fold_test, reader=reader)

    # Algorithm
    # algo = NMF
    algo = CoClustering(n_cltr_u=n_cltr_u, n_cltr_i=n_cltr_i,
                        n_epochs=n_epochs, verbose=True)

    for trainset, valset in data_val.folds():
        # Train
        algo.fit(trainset)
        # Validation
        valpred = algo.test(valset)

    for trainset, testset in data_test.folds():
        testpred = algo.test(testset)

    # Clipping:
    tpred = np.zeros(len(testpred))
    vpred = np.zeros(len(valpred))
    for i in range(len(testpred)):
        val = testpred[i].est
        if val > 5:
            tpred[i] = 5
        elif val < 1:
            tpred[i] = 1
        else:
            tpred[i] = val

    for i in range(len(valpred)):
        val = valpred[i].est
        if val > 5:
            vpred[i] = 5
        elif val < 1:
            vpred[i] = 1
        else:
            vpred[i] = val

    # Copy the test
    test_df = test.copy()
    test_df.Rating = tpred

    val_df = validation.copy()
    val_df.Rating = vpred


    val_submission = submission_table(val_df, 'User', 'Movie', 'Rating')
    test_submission = submission_table(test_df, 'User', 'Movie', 'Rating')

    val_fn = 'cache/' + fn
    test_fn = 'test/' + fn

    print("Writing test file:")
    test_submission.to_csv(test_fn, index=False)
    print("Writing validation file:")
    val_submission.to_csv(val_fn, index=False)
def slopeOne(train, validation, test, fn):
    """
    slopeOne from library Surprise
    """
    # Get parameters

    # First, we need to dump the pandas DF into files
    train_file = 'tmp_train.csv'
    test_file = 'tmp_test.csv'
    val_file = 'tmp_val.csv'
    train.to_csv(train_file, index=False, header=False)
    test.to_csv(test_file, index=False, header=False)
    validation.to_csv(val_file, index=False, header=False)

    # Create Reader
    reader = Reader(line_format='user item rating', sep=',')

    # Train and test set for Surprise
    fold_val = [(train_file, val_file)]
    fold_test = [(train_file, test_file)]

    # Load the data
    data_val = Dataset.load_from_folds(fold_val, reader=reader)
    data_test = Dataset.load_from_folds(fold_test, reader=reader)

    # Algorithm
    # algo = SVDpp(sim_options=sim_options)
    algo = SlopeOne()

    for trainset, valset in data_val.folds():
        # Train
        algo.fit(trainset)
        # Validation
        valpred = algo.test(valset)

    for trainset, testset in data_test.folds():
        testpred = algo.test(testset)

    # Clipping:
    tpred = np.zeros(len(testpred))
    vpred = np.zeros(len(valpred))
    for i in range(len(testpred)):
        val = testpred[i].est
        if val > 5:
            tpred[i] = 5
        elif val < 1:
            tpred[i] = 1
        else:
            tpred[i] = val

    for i in range(len(valpred)):
        val = valpred[i].est
        if val > 5:
            vpred[i] = 5
        elif val < 1:
            vpred[i] = 1
        else:
            vpred[i] = val

    # Copy the test
    test_df = test.copy()
    test_df.Rating = tpred

    val_df = validation.copy()
    val_df.Rating = vpred


    val_submission = submission_table(val_df, 'User', 'Movie', 'Rating')
    test_submission = submission_table(test_df, 'User', 'Movie', 'Rating')

    val_fn = 'cache/' + fn
    test_fn = 'test/' + fn

    print("Writing test file:")
    test_submission.to_csv(test_fn, index=False)
    print("Writing validation file:")
    val_submission.to_csv(val_fn, index=False)


def KNNmain(item=True, user=True):
    train = load_csv('data/trainTruth.csv')
    val = load_csv('data/valTruth.csv')
    test = load_csv('data/sampleSubmission.csv')

    if item:
        name = 'KNN_item'
        knn(train, val, test, name, k=60, sim_options={'name': 'pearson_baseline', 'user_based': False})
    if user:
        name = 'KNN_user'
        knn(train, val, test, name, k=60, sim_options={'name': 'pearson_baseline', 'user_based': True})

def surprise_model(item=True, user=True, slope=True, svdp=True, nmf=True, coclustering=True):
    train = load_csv('data/trainTruth.csv')
    val = load_csv('data/valTruth.csv')
    test = load_csv('data/sampleSubmission.csv')

    if item:
        name = 'KNN_item'
        knn(train, val, test, name, k=60, sim_options={'name': 'pearson_baseline', 'user_based': False})
    if user:
        name = 'KNN_user'
        knn(train, val, test, name, k=60, sim_options={'name': 'pearson_baseline', 'user_based': True})
    if slope:
        name = 'slopeOne'
        slopeOne(train, val, test, name)
    if svdp:
        name = 'SVDpp'
        SVDsuprise(train, val, test, name, n_factors = 10, n_epochs = 10, reg_all = 0.01, lr_all=0.01)
    if nmf:
        name = 'NMF'
        NMFsuprise(train, val, test, name, n_factors = 15, n_epochs = 100, biased=True, reg_pu=0.06, reg_qi=0.06, reg_bu=0.02,
                   reg_bi=0.02, lr_bu=0.005, lr_bi=0.005, init_low=0, init_high=1)
    if coclustering:
        name = "CoClustering"
        CoClusteringsuprise(train, val, test, name, n_cltr_u=3, n_cltr_i=3,
                            n_epochs=50)


if __name__ == "__main__":
    train = load_csv('data/trainTruth.csv')
    val = load_csv('data/valTruth.csv')
    test = load_csv('data/sampleSubmission.csv')
    # name = 'KNN_item'
    # knn(train, val, test, name, k=60, sim_options={'name': 'pearson_baseline', 'user_based': False})
    #name = 'SVDpp'
    #SVDsuprise(train, val, test, name, n_factors = 10, n_epochs = 5, reg_all = 0.01, lr_all=0.01)
    #name = 'SVDpp'
    #SVDsuprise(train, val, test, name, n_factors = 10, n_epochs = 80, reg_all = 0.05, lr_all=0.005)
    # name = 'slopeOne'
    # slopeOne(train, val, test, name)
    #name = 'NMF'
    #NMFsuprise(train, val, test, name, n_factors = 15, n_epochs = 100, biased=True, reg_pu=0.06, reg_qi=0.06, reg_bu=0.02,
    #           reg_bi=0.02, lr_bu=0.005, lr_bi=0.005, init_low=0, init_high=1)
    name = "CoClustering"
    CoClusteringsuprise(train, val, test, name, n_cltr_u=3, n_cltr_i=3,
                        n_epochs=50)
