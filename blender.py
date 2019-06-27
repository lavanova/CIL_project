import scipy.optimize as sco
import numpy as np
from utils import *
from os import listdir
from os.path import isfile, join
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run Blender.")
    parser.add_argument('--mode', type=int, default=0,
                        help='0: blend all models;  1: blend specified models')

    parser.add_argument('--models', nargs='?', default='["KNN_item", "svd_col"]')

    return parser.parse_args()


def eval_(x, pred_dic, trueval):
    values = np.stack([pred_dic[i] for i in pred_dic], axis=0)
    x = x.reshape([len(pred_dic), 1])
    values = np.sum(x*values, axis=0)
    return np.sqrt(np.mean(np.square(values - trueval)))


def blender(args):
    trueval = LoadRawData(parameters.VALTRUTH_PATH)
    csvdir = 'cache/'
    if args.mode == 0:
        model_names = [f for f in listdir(csvdir) if isfile(join(csvdir, f))]
    elif args.mode == 1:
        model_names = args.models
    pred_dic = {}
    weight_dic = {}
    for model in model_names:
        pred_dic[model] = LoadRawData(csvdir+model)
        mse = np.mean(np.square(trueval - pred_dic[model]))
        rmse = np.sqrt(np.mean(np.square(trueval - pred_dic[model])))
        print(model+' rmse: '+str(rmse)+'   mse: '+str(mse))
    assert(len(model_names) == len(pred_dic))
    initval = np.ones(len(model_names)) / len(model_names)
    res = sco.minimize(eval_, initval, method='SLSQP', args=(pred_dic, trueval), options={'maxiter':1000, 'disp':True})
    print(res.x)
    pt = 0
    for i in pred_dic:
        weight_dic[i] = res.x[pt]
        pt += 1
    return weight_dic

def checkdist():
    csvdir = 'cache/'
    model_names = [f for f in listdir(csvdir) if isfile(join(csvdir, f))]
    pred_dic = {}
    for model in model_names:
        pred_dic[model] = LoadRawData(csvdir+model)
    assert(len(model_names) == len(pred_dic))
    values = np.stack([pred_dic[i] for i in pred_dic], axis=0)
    avgvar = np.mean(np.std(values, axis=0))
    return avgvar


def applyblender(weight_dic, opath=parameters.OUTPUTCSV_PATH, args=None):
    testdir = 'test/'
    if args.mode == 0:
        model_names = [f for f in listdir(testdir) if isfile(join(testdir, f))]
    elif args.mode == 1:
        model_names = args.models
    result = 0
    for model in model_names:
        assert (model in weight_dic), "ApplyBlender: test csv key not found"
        result += weight_dic[model] * LoadRawData(testdir + model)
    sample = parameters.SAMPLECSV_PATH
    template = pd.read_csv(sample)
    rcstrs = template.values[:,0]
    df = pd.DataFrame({'Id': rcstrs, 'Prediction': result})
    df.to_csv(opath, index=False)


# def WriteToCSV(data, path = parameters.OUTPUTCSV_PATH, sample = parameters.SAMPLECSV_PATH): # expect the input to be a matrix
#     print("writing result to csv")
#     template = pd.read_csv(sample)
#     size = template.values.shape[0]
#     rcstrs = [None] * size
#     values = [0] * size
#     count = 0
#     for i in template.values:
#         r, c = GetRC(i[0])
#         rcstrs[count] = i[0]
#     df = pd.DataFrame({'Id': rcstrs,'Prediction': values})
#     df.to_csv(path,index=False)
#     print("writing completed")

if __name__ == '__main__':
    # print(checkdist())
    args = parse_args()
    print(type(args.models))
    print(args.models)
    args.models = eval(args.models)
    weight_dic = blender(args)
    print(weight_dic)
    applyblender(weight_dic, args=args)
