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

if __name__ == "__main__":
    train = load_csv('data/trainTruth.csv')
    val = load_csv('data/valTruth.csv')
    test = load_csv('data/sampleSubmission.csv')
    name = 'KNN_item'
    knn(train, val, test, name, k=60, sim_options={'name': 'pearson_baseline', 'user_based': False})