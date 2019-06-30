import tensorflow as tf
#Numpy contains helpful functions for efficient mathematical calculations
import numpy as np
#Dataframe manipulation library
import pandas as pd
import csv
import copy
from scipy.sparse import csr_matrix
import random
N_IT = 5
ETA = 0.001
users = {}
user_movies = {}
#data = {}
raw_data={}
def GetRC(s):
    s = s.strip('r')
    l = s.split('_c')
    r = int(l[0]) - 1 #index starts at 0, while in csv it starts at 1
    c = int(l[1]) - 1
    return r,c


def LoadTrainValDataMask(valper=0.15, inpath = "data_train.csv"):
    rawdata = pd.read_csv(url)
    train_users={}
    val_users={}
    train_user_map = {}
    train_user_ct = 0
    val_user_map = {}
    val_user_ct = 0
    #train_data = np.zeros( (10000, 1000), dtype=np.float32 )
    train_mask = np.zeros( (10000, 1000), dtype=np.int )
    #val_data = np.zeros( (10000, 1000), dtype=np.float32 )
    val_mask = np.zeros( (10000, 1000), dtype=np.int )
    for i in rawdata.values:
        r, c = GetRC(i[0])
        dice = random.uniform(0, 1)
        
        
        if dice < valper:
            if r not in val_user_map:
                val_user_map[r] = val_user_ct
                val_user_ct += 1
            u = val_user_map[r]
            if u in val_users:
                val_users[u].append((c, i[1]))
            else:
                val_users[u] = []
                val_users[u].append((c, i[1]))
            #val_data[r,c] = (i[1])
            val_mask[r,c] = 1
        else:
            if r not in train_user_map:
                train_user_map[r] = train_user_ct
                train_user_ct += 1
            u = train_user_map[r]
            if u in train_users:
                train_users[u].append((c, i[1]))
            else:
                train_users[u] = []
                train_users[u].append((c, i[1]))
            train_mask[r,c] = 1
    #data,mean,std=normalizeDataMask(data, mask)
    print("Load train validation data mask complete")
    return train_users, train_user_map, val_users, val_user_map

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
class RBM():

    def __init__(self, data):
        
        self.F = 50
        self.K = 5
        self.m = 0
        #self.m = data.shape[0]   # No of movies rated by user
        for i, u in enumerate(users):
            temp = [movie_id for (movie_id, rating) in raw_data[u]]
            self.m = max(self.m, max(i for i in temp) + 1)
        print (self.m)
        self.h = np.random.rand(self.F) - 0.5
        self.featureBias = np.random.rand(self.F) - 0.5
        self.movieBias = np.random.rand(self.m, self.K) - 0.5
        self.w = np.random.rand(self.F, self.m, self.K) - 0.5
        self.data = data
        

    def train(self):
        for it in range(N_IT):
            error=0
            for u in users:
                data = copy.deepcopy(self.data[u])
                w = self.getW(user_movies[u])
                posAssociations, self.h = self.fwdProp(data, user_movies[u])
                visibleProb = self.bwdProp(self.h, user_movies[u])
                negAssociations, temp = self.fwdProp(visibleProb, user_movies[u])
                w += ETA * (posAssociations - negAssociations) / len(user_movies[u]) #might change len
                self.setW(user_movies[u], w)
                error_ = np.sum((data - visibleProb) ** 2)
                error += np.sqrt(error_/len(data))
                print(np.sqrt(error_/len(data)))
            print (it, error/1000)


    def getW(self, movies):

        a = np.zeros((self.F, 1, self.K))
        for m in movies:
            a = np.concatenate((a, np.expand_dims(self.w[:,m,:], axis=1)), axis=1)
        return a[:,1:,]


    def setW(self, movies, w):
        
        it = 0
        for m in movies:
            self.w[:, m, :] = w[:, it, :]
            it += 1
        
    
    def getMovieBias(self, movies):
        
        a = np.zeros((1, self.K))
        for m in movies:
            a = np.concatenate((a, np.expand_dims(self.movieBias[m,:], axis=0)), axis=0)
        return a[1:,]


    def fwdProp(self, inp, movies):
        hiddenUnit = np.copy(self.featureBias)
        for j in range(self.F):
            hiddenUnit[j] += np.tensordot(inp, self.getW(movies)[j])
        hiddenProb = sigmoid(hiddenUnit)
        hiddenStates = hiddenProb > np.random.rand(self.F)
        hiddenAssociations = np.zeros((self.F, len(movies), self.K))    # Same as self.w for a single user case
        for j in range(self.F):
            hiddenAssociations[j] = hiddenProb[j] * inp
        return hiddenAssociations, hiddenStates


    def bwdProp(self, inp, movies):
        visibleUnit = self.getMovieBias(movies)
        for j in range(self.F):
            visibleUnit += inp[j] * self.getW(movies)[j]
        visibleProb = sigmoid(visibleUnit)
        return visibleProb


    def predictor(self, movie_id, user_id):

        w = self.getW(user_movies[user_id])
        
        #making predictions part Vq not given
        data = copy.deepcopy(self.data[user_id])
        probs = np.ones(5)
        
        mx, index = -1, 0

        for i in range(5):
            calc = 1.0
            for j in range(self.F):
                temp = np.tensordot(data, self.getW(user_movies[user_id])[j]) + self.featureBias[j]
                temp = 1.0 + np.exp(temp)
                calc *= temp
            probs[i] = calc

            if mx < probs[i]:
                index = i
                mx = probs[i]

        return index

if __name__ == "__main__":
    raw_data1,train_user_map,val_data1,val_user_map = LoadTrainValDataMask()
    ct = 0
    for i in raw_data1.keys():
        raw_data[i] = raw_data1[i]
        ct += 1
        if ct >= 1000:
            break
    data = {}
    users = raw_data.keys()
    for i, u in enumerate(users):
        #user_movies[u] = [movie_id for (movie_id, rating) in raw_data1[u]]
        data[u] = [[0]*(rat-1) + [1] + [0]*(5-rat) for (mov_id, rat) in raw_data[u]]
        data[u] = np.asarray(data[u])
    for i, u in enumerate(raw_data1.keys()):
        user_movies[u] = [movie_id for (movie_id, rating) in raw_data1[u]]
    rbm = RBM(data)
    rbm.train()
    print("writing result to csv")
    template = pd.read_csv('sampleSubmission.csv')
    size = template.values.shape[0]
    rcstrs = [None] * size
    values = [0] * size
    count = 0
    for i in template.values:
        r, c = GetRC(i[0])
        rcstrs[count] = i[0]
        try:
            values[count] = rbm.predictor(c,train_user_map[r])
            #print(" found")
        except:
            values[count]=2.5
        #rating=rbm.predictor(c,train_user_map[r])
        count += 1
        # data frame is reconstructed since the direct modification is too slow
    df = pd.DataFrame({'Id': rcstrs,'Prediction': values})
    df.to_csv('rbm.csv',index=False)
    print("writing completed")

    