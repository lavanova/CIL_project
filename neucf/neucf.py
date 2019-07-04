import numpy as np
import tensorflow as tf
from data import create_dataloader_train, create_dataloader_test
import argparse
from model import NeuCF, NeuCF2, NeuCF3, NeuCF4
import os
import math
import pandas as pd
from tqdm import tqdm
from utils import early_stopping
from shutil import copyfile

def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--mode', type=int, default=0,
                        help='0: training; 1: inference on valid set and test set with pretrained model')
    parser.add_argument('--model_path', nargs='?', default='',
                        help='load path of pretrained model')     
    parser.add_argument('--test_path', nargs='?', default='',
                        help='when in mode 1, path of output of test')
    parser.add_argument('--output_valid_path', nargs='?', default='',
                        help='when in mode 1, path of output of output_valid') 
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--log_path', nargs='?', default='log/model/',
                        help='log path')
    parser.add_argument('--save_flag', type=bool, default=True,
                        help='whether save')
    parser.add_argument('--flag_step', type=int, default=100,
                        help='flag step')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--model', nargs='?', default='NeuCF2',
                        help='which model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--epoch_iter', type=int, default=4597,
                        help='how many batches in a epoch')
    parser.add_argument('--valid_iter', type=int, default=100,
                        help='how many batches are used to do validation every time')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='valid train set split ratio')
    parser.add_argument('--external_embedding', type=bool, default=False,
                        help='whether use external embeddings')
    parser.add_argument('--external_embedding_type', type=int, default=0,
                        help='0: lle, spectral, nmf, factor; 1: neural graph embedding')
    parser.add_argument('--external_embedding_trainable', type=bool, default=False,
                        help='whether external embedding is trainable or not')
    parser.add_argument('--graph_embedding_dim', type=int, default=256,
                        help='dimension of graph embedding')
    parser.add_argument('--graph_embedding_row_path', nargs='?', default='',
                        help='load path to graph embedding of row')
    parser.add_argument('--graph_embedding_col_path', nargs='?', default='',
                        help='load path to graph embedding of col')
    parser.add_argument('--graph_embedding_scale', nargs='?', default='[3.45, 5.57]',
                        help='scale factor for graph embedding')
    parser.add_argument('--loss_type', nargs='?', default='mse',
                        help='loss type:mse, cross_entropy,...')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout probability')
    parser.add_argument('--decay_step', type=int, default=1500,
                        help='decay step')
    parser.add_argument('--decay_rate', type=float, default=0.96,
                        help='decay rate')
    parser.add_argument('--batch_norm', type=bool, default=True,
                        help='whether use batch normalization')
    parser.add_argument('--max_norm_res', type=bool, default=True,
                        help='whether use max_norm in residual block')
    parser.add_argument('--residual', type=bool, default=False,
                        help='whether add residual connection')
    parser.add_argument('--residual_block', type=int, default=2,
                        help='how many residual blocks')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')                    
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()

def _train(args):
    os.makedirs(os.path.join(args.log_path), exist_ok=True)
    logfile = open(os.path.join(args.log_path, 'log.txt'), 'w', buffering=1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    with tf.Session(config=config) as sess:
        dataloader_train, dataloader_valid, max_row, max_col, iterator_output, rcstrs_output, iterator_output_valid, rcstrs_output_valid = create_dataloader_train(valid_ratio=args.valid_ratio, batch_size=args.batch_size)
        #dataloader_test, row_col_prediction, rcstrs = create_dataloader_test(batch_size=args.batch_size)
        iterator_test, row_col_prediction, rcstrs = create_dataloader_test(batch_size=args.batch_size)
        sess.run([iterator_test.initializer,
                  iterator_output.initializer,
                  iterator_output_valid.initializer])
        dataloader_test = iterator_test.get_next()
        dataloader_output = iterator_output.get_next()
        dataloader_output_valid = iterator_output_valid.get_next()
        row_col_train, label_train = dataloader_train
        row_col_valid, label_valid = dataloader_valid
        row_col_output, label_output = dataloader_output
        row_col_output_valid, label_output_valid = dataloader_output_valid
        row_col_test, label_test = dataloader_test
        if args.model == "NeuCF2":
            with tf.variable_scope("model", reuse=False):
                model_train = NeuCF2(row_col_train, label_train, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())

            with tf.variable_scope("model", reuse=True):
                model_valid = NeuCF2(row_col_valid, label_valid, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())

            with tf.variable_scope("model", reuse=True):
                model_test = NeuCF2(row_col_test, label_test, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())
            with tf.variable_scope("model", reuse=True):
                model_output = NeuCF2(row_col_output, label_output, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())    
            with tf.variable_scope("model", reuse=True):
                model_output_valid = NeuCF2(row_col_output_valid, label_output_valid, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())

            if args.external_embedding:
                model_train.init_embedding(sess, args)
                #model_valid.init_embedding(sess, args)
                #model_test.init_embedding(sess, args)
        elif args.model == "NeuCF3":
            with tf.variable_scope("model", reuse=False):
                model_train = NeuCF3(row_col_train, label_train, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())

            with tf.variable_scope("model", reuse=True):
                model_valid = NeuCF3(row_col_valid, label_valid, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())

            with tf.variable_scope("model", reuse=True):
                model_test = NeuCF3(row_col_test, label_test, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())

            with tf.variable_scope("model", reuse=True):
                model_output = NeuCF3(row_col_output, label_output, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())    
            with tf.variable_scope("model", reuse=True):
                model_output_valid = NeuCF3(row_col_output_valid, label_output_valid, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())

            if args.external_embedding:
                model_train.init_embedding(sess, args)     
        elif args.model == "NeuCF4":
            with tf.variable_scope("model", reuse=False):
                model_train = NeuCF4(row_col_train, label_train, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())

            with tf.variable_scope("model", reuse=True):
                model_valid = NeuCF4(row_col_valid, label_valid, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())

            with tf.variable_scope("model", reuse=True):
                model_test = NeuCF4(row_col_test, label_test, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())

            with tf.variable_scope("model", reuse=True):
                model_output = NeuCF4(row_col_output, label_output, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())    
            with tf.variable_scope("model", reuse=True):
                model_output_valid = NeuCF4(row_col_output_valid, label_output_valid, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())
            if args.external_embedding:
                model_train.init_embedding(sess, args)                   
        elif args.model == "NeuCF":
            with tf.variable_scope("model", reuse=False):
                model_train = NeuCF(row_col_train, label_train, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())

            with tf.variable_scope("model", reuse=True):
                model_valid = NeuCF(row_col_valid, label_valid, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())

            with tf.variable_scope("model", reuse=True):
                model_test = NeuCF(row_col_test, label_test, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())           

            with tf.variable_scope("model", reuse=True):
                model_output = NeuCF(row_col_output, label_output, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())    
            with tf.variable_scope("model", reuse=True):
                model_output_valid = NeuCF(row_col_output_valid, label_output_valid, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(args.log_path, sess.graph)

        vars = [v for v in tf.global_variables() if v.name.startswith("model/NeuCF")]
        saver = tf.train.Saver(vars, max_to_keep=200)
        should_stop = False
        stopping_step = 0
        cur_best_pre_0 = 1.0
        for i in range(args.epochs):
            epoch_loss = 0
            for j in range(args.epoch_iter):
                batch_loss = model_train.step(sess, isTraining=True, dropout_keep_prob=0.5)
                epoch_loss += batch_loss / args.epoch_iter
            valid_loss = 0
            valid_sse = 0
            for j in range(args.valid_iter):
                batch_loss, batch_sse = model_valid.step(sess, isValidating=True, dropout_keep_prob=1)
                valid_loss += batch_loss / args.valid_iter
                valid_sse += batch_sse
            valid_sse = valid_sse / (args.valid_iter * args.batch_size)
            valid_rmse = np.sqrt(valid_sse)
            print( '--Avg. Train Loss ='+str(epoch_loss)[:6] + '    --Avg. Valid Loss ='+str(valid_loss)[:6]+ '    --Valid RMSE = '+str(valid_rmse)[:6]+'\n' )
            logfile.write( '--Avg. Train Loss ='+str(epoch_loss)[:6] + '    --Avg. Valid Loss ='+str(valid_loss)[:6]+ '    --Valid RMSE = '+str(valid_rmse)[:6]+'\n' )
            logfile.flush()
            cur_best_pre_0, stopping_step, should_stop = early_stopping(valid_rmse, cur_best_pre_0, 
                                                                        stopping_step, expected_order='dec',
                                                                        flag_step=args.flag_step)
            if should_stop == True:
                break
            #saver.save(sess, os.path.join(args.log_path,'model'), global_step=i, write_meta_graph=False)
            if valid_rmse == cur_best_pre_0 and args.save_flag:
                saver.save(sess, os.path.join(args.log_path,'model'), global_step=i+1, write_meta_graph=False)
                output_prediction = None
                for j in range( math.ceil( len(rcstrs_output) / args.batch_size) ):
                    predict = model_output.step(sess, isTesting=True, dropout_keep_prob=1)
                    if j == 0:
                        output_prediction = predict
                    else:
                        output_prediction = np.concatenate([output_prediction, predict], axis=0)
                output_prediction = np.reshape(output_prediction, (output_prediction.shape[0],))
                df = pd.DataFrame( {'Id': rcstrs_output,'Prediction': output_prediction} )
                df.to_csv(os.path.join(args.log_path, 'output' + str(i+1)+".csv" ),index=False)
                #copyfile( os.path.join(args.log_path, 'output' + str(i+1)+".csv" ) , "../" )
                output_valid_prediction = None
                for j in range( math.ceil( len(rcstrs_output_valid) / args.batch_size) ):
                    predict = model_output_valid.step(sess, isTesting=True, dropout_keep_prob=1)
                    if j == 0:
                        output_valid_prediction = predict
                    else:
                        output_valid_prediction = np.concatenate( [output_valid_prediction, predict] , axis=0 )
                output_valid_prediction = np.reshape(output_valid_prediction, (output_valid_prediction.shape[0],))
                df = pd.DataFrame( {'Id': rcstrs_output_valid,'Prediction': output_valid_prediction} )
                df.to_csv(os.path.join(args.log_path, 'output_valid' + str(i+1)+".csv" ),index=False)
                copyfile( os.path.join(args.log_path, 'output_valid' + str(i+1)+".csv" ) , 
                                       os.path.join("../cache", args.log_path.split("/")[-2] ) )
                test_prediction = None
                for j in range(math.ceil(row_col_prediction.shape[0] / args.batch_size)):
                    predict = model_test.step(sess, isTesting=True, dropout_keep_prob=1)
                    if j == 0:
                        test_prediction = predict
                    else:
                        test_prediction = np.concatenate([test_prediction, predict], axis=0)
                test_prediction = np.reshape(test_prediction, (test_prediction.shape[0],))
                
                # data frame is reconstructed since the direct modification is too slow
                df = pd.DataFrame({'Id': rcstrs,'Prediction': test_prediction})
                df.to_csv(os.path.join(args.log_path, 'submission' + str(i+1)+".csv" ),index=False)
                copyfile( os.path.join(args.log_path, 'submission' + str(i+1)+".csv" ) , 
                                       os.path.join("../test", args.log_path.split("/")[-2] ) )
                sess.run([iterator_test.initializer,
                        iterator_output.initializer,
                        iterator_output_valid.initializer])
                dataloader_test = iterator_test.get_next()
                dataloader_output = iterator_output.get_next()
                dataloader_output_valid = iterator_output_valid.get_next()
                row_col_test, label_test = dataloader_test
                row_col_output, label_output = dataloader_output
                row_col_output_valid, label_output_valid = dataloader_output_valid

def _inference(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    with tf.Session(config=config) as sess:
        dataloader_train, dataloader_valid, max_row, max_col, iterator_output, rcstrs_output, iterator_output_valid, rcstrs_output_valid = create_dataloader_train(valid_ratio=args.valid_ratio, batch_size=args.batch_size)
        #dataloader_test, row_col_prediction, rcstrs = create_dataloader_test(batch_size=args.batch_size)
        iterator_test, row_col_prediction, rcstrs = create_dataloader_test(batch_size=args.batch_size)
        sess.run([iterator_test.initializer,
                  #iterator_output.initializer,
                  iterator_output_valid.initializer])
        dataloader_test = iterator_test.get_next()
        #dataloader_output = iterator_output.get_next()
        dataloader_output_valid = iterator_output_valid.get_next()
        #row_col_train, label_train = dataloader_train
        #row_col_valid, label_valid = dataloader_valid
        #row_col_output, label_output = dataloader_output
        row_col_output_valid, label_output_valid = dataloader_output_valid
        row_col_test, label_test = dataloader_test
        if args.model == "NeuCF2":
            #with tf.variable_scope("model", reuse=False):
            #    model_train = NeuCF2(row_col_train, label_train, max_row, max_col, args)
            #    sess.run(tf.global_variables_initializer())

            #with tf.variable_scope("model", reuse=True):
            #    model_valid = NeuCF2(row_col_valid, label_valid, max_row, max_col, args)
            #    sess.run(tf.global_variables_initializer())

            with tf.variable_scope("model", reuse=False):
                model_test = NeuCF2(row_col_test, label_test, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())
            #with tf.variable_scope("model", reuse=True):
            #    model_output = NeuCF2(row_col_output, label_output, max_row, max_col, args)
            #    sess.run(tf.global_variables_initializer())    
            with tf.variable_scope("model", reuse=True):
                model_output_valid = NeuCF2(row_col_output_valid, label_output_valid, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())

            if args.external_embedding:
                model_test.init_embedding(sess, args)
                #model_valid.init_embedding(sess, args)
                #model_test.init_embedding(sess, args)
        elif args.model == "NeuCF3":
            #with tf.variable_scope("model", reuse=False):
            #    model_train = NeuCF3(row_col_train, label_train, max_row, max_col, args)
            #    sess.run(tf.global_variables_initializer())

            #with tf.variable_scope("model", reuse=True):
            #    model_valid = NeuCF3(row_col_valid, label_valid, max_row, max_col, args)
            #    sess.run(tf.global_variables_initializer())

            with tf.variable_scope("model", reuse=False):
                model_test = NeuCF3(row_col_test, label_test, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())

            #with tf.variable_scope("model", reuse=True):
            #    model_output = NeuCF3(row_col_output, label_output, max_row, max_col, args)
            #    sess.run(tf.global_variables_initializer())    
            with tf.variable_scope("model", reuse=True):
                model_output_valid = NeuCF3(row_col_output_valid, label_output_valid, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())

            if args.external_embedding:
                model_test.init_embedding(sess, args)     
        elif args.model == "NeuCF4":
            #with tf.variable_scope("model", reuse=False):
            #    model_train = NeuCF4(row_col_train, label_train, max_row, max_col, args)
            #    sess.run(tf.global_variables_initializer())

            #with tf.variable_scope("model", reuse=True):
            #    model_valid = NeuCF4(row_col_valid, label_valid, max_row, max_col, args)
            #    sess.run(tf.global_variables_initializer())

            with tf.variable_scope("model", reuse=False):
                model_test = NeuCF4(row_col_test, label_test, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())

            #with tf.variable_scope("model", reuse=True):
            #    model_output = NeuCF4(row_col_output, label_output, max_row, max_col, args)
            #    sess.run(tf.global_variables_initializer())    
            with tf.variable_scope("model", reuse=True):
                model_output_valid = NeuCF4(row_col_output_valid, label_output_valid, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())
            if args.external_embedding:
                model_test.init_embedding(sess, args)                   
        elif args.model == "NeuCF":
            #with tf.variable_scope("model", reuse=False):
            #    model_train = NeuCF(row_col_train, label_train, max_row, max_col, args)
            #    sess.run(tf.global_variables_initializer())

            #with tf.variable_scope("model", reuse=True):
            #    model_valid = NeuCF(row_col_valid, label_valid, max_row, max_col, args)
            #    sess.run(tf.global_variables_initializer())

            with tf.variable_scope("model", reuse=False):
                model_test = NeuCF(row_col_test, label_test, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())           

            #with tf.variable_scope("model", reuse=True):
            #    model_output = NeuCF(row_col_output, label_output, max_row, max_col, args)
            #    sess.run(tf.global_variables_initializer())    
            with tf.variable_scope("model", reuse=True):
                model_output_valid = NeuCF(row_col_output_valid, label_output_valid, max_row, max_col, args)
                sess.run(tf.global_variables_initializer())
        vars = [v for v in tf.global_variables() if v.name.startswith("model/NeuCF")]
        saver = tf.train.Saver(vars)
        saver.restore(sess, args.model_path)

        output_valid_prediction = None
        for j in range( math.ceil( len(rcstrs_output_valid) / args.batch_size) ):
            predict = model_output_valid.step(sess, isTesting=True, dropout_keep_prob=1)
            if j == 0:
                output_valid_prediction = predict
            else:
                output_valid_prediction = np.concatenate( [output_valid_prediction, predict] , axis=0 )
        output_valid_prediction = np.reshape(output_valid_prediction, (output_valid_prediction.shape[0],))
        df = pd.DataFrame( {'Id': rcstrs_output_valid,'Prediction': output_valid_prediction} )
        df.to_csv(args.output_valid_path,index=False)
        test_prediction = None
        for j in range(math.ceil(row_col_prediction.shape[0] / args.batch_size)):
            predict = model_test.step(sess, isTesting=True, dropout_keep_prob=1)
            if j == 0:
                test_prediction = predict
            else:
                test_prediction = np.concatenate([test_prediction, predict], axis=0)
        test_prediction = np.reshape(test_prediction, (test_prediction.shape[0],))
        
        # data frame is reconstructed since the direct modification is too slow
        df = pd.DataFrame({'Id': rcstrs,'Prediction': test_prediction})
        df.to_csv(args.test_path,index=False)

if __name__ == '__main__':
    args = parse_args()
    args.layers = eval(args.layers)
    args.reg_layers = eval(args.reg_layers)
    args.graph_embedding_scale = eval(args.graph_embedding_scale)
    if args.mode == 0:
        _train(args)
    elif args.mode == 1:
        _inference(args)
