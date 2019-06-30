from utils import *
def WriteToCSV_movie(data, path = parameters.OUTPUTCSV_PATH, sample = parameters.SAMPLECSV_PATH): # expect the input to be a matrix
    print("writing result to csv")
    template = pd.read_csv(sample)
    size = template.values.shape[0]
    rcstrs = [None] * size
    values = [0] * size
    count = 0
    for i in template.values:
        r, c = GetRC(i[0])
        rcstrs[count] = i[0]
        values[count] = data[c]
        count += 1
    # data frame is reconstructed since the direct modification is too slow
    df = pd.DataFrame({'Id': rcstrs,'Prediction': values})
    df.to_csv(path,index=False)
    print("writing completed")
    return values
def WriteToCSV_user(data, path = parameters.OUTPUTCSV_PATH, sample = parameters.SAMPLECSV_PATH): # expect the input to be a matrix
    print("writing result to csv")
    template = pd.read_csv(sample)
    size = template.values.shape[0]
    rcstrs = [None] * size
    values = [0] * size
    count = 0
    for i in template.values:
        r, c = GetRC(i[0])
        rcstrs[count] = i[0]
        values[count] = data[r]
        count += 1
    # data frame is reconstructed since the direct modification is too slow
    df = pd.DataFrame({'Id': rcstrs,'Prediction': values})
    df.to_csv(path,index=False)
    print("writing completed")
    return values
def WriteToCSV_both(data_user,data_movie, path = parameters.OUTPUTCSV_PATH, sample = parameters.SAMPLECSV_PATH): # expect the input to be a matrix
    print("writing result to csv")
    template = pd.read_csv(sample)
    size = template.values.shape[0]
    rcstrs = [None] * size
    values = [0] * size
    count = 0
    for i in template.values:
        r, c = GetRC(i[0])
        rcstrs[count] = i[0]
        values[count] = (data_user[r]+data_movie[c])/2
        count += 1
    # data frame is reconstructed since the direct modification is too slow
    df = pd.DataFrame({'Id': rcstrs,'Prediction': values})
    df.to_csv(path,index=False)
    print("writing completed")
    return values


def main():
    train_data, train_mask, val_data, val_mask=LoadFixedValDataMask()
    movie_mean=np.sum(train_data, axis=0)/np.sum((train_data!=0),axis=0)
    user_mean=np.sum(train_data, axis=1)/np.sum((train_data!=0),axis=1)
    value_movie=WriteToCSV_movie(movie_mean,path='cache/movie_mean', sample=parameters.VALTRUTH_PATH)
    _=WriteToCSV_movie(movie_mean,path='test/movie_mean', sample=parameters.SAMPLECSV_PATH)
    value_user=WriteToCSV_user(user_mean,path='cache/user_mean', sample=parameters.VALTRUTH_PATH)
    _=WriteToCSV_user(user_mean,path='test/user_mean', sample=parameters.SAMPLECSV_PATH)
    value_both=WriteToCSV_both(user_mean,movie_mean,path='cache/both_mean', sample=parameters.VALTRUTH_PATH)
    _=WriteToCSV_both(user_mean,movie_mean,path='test/both_mean', sample=parameters.SAMPLECSV_PATH)
    trueval = LoadRawData(parameters.VALTRUTH_PATH)
    mse = np.mean(np.square(trueval - value_movie))
    rmse = np.sqrt(np.mean(np.square(trueval - value_movie)))
    print("movie"+' rmse: '+str(rmse)+'   mse: '+str(mse))
    mse = np.mean(np.square(trueval - value_user))
    rmse = np.sqrt(np.mean(np.square(trueval - value_user)))
    print("user"+' rmse: '+str(rmse)+'   mse: '+str(mse))
    mse = np.mean(np.square(trueval - value_both))
    rmse = np.sqrt(np.mean(np.square(trueval - value_both)))
    print("both"+' rmse: '+str(rmse)+'   mse: '+str(mse))





if __name__ == "__main__":
    main()