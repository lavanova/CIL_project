import numpy as np
import tensorflow as tf
from data import create_dataloader_train
import argparse
from model import NeuCF
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--log_path', nargs='?', default='log/model/',
                        help='log path')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
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
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
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
    logfile = open(os.path.join(args.log_path, 'log.txt'), 'w', buffering=1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    with tf.Session(config=config) as sess:
        dataloader_train, dataloader_valid, max_row, max_col = create_dataloader_train(valid_ratio=args.valid_ratio, batch_size=args.batch_size)
        row_col_train, label_train = dataloader_train
        row_col_valid, label_valid = dataloader_valid

        with tf.variable_scope("model", reuse=False):
            model_train = NeuCF(row_col_train, label_train, max_row, max_col, args)
            sess.run(tf.global_variables_initializer())

        with tf.variable_scope("model", reuse=True):
            model_valid = NeuCF(row_col_valid, label_valid, max_row, max_col, args)
            sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(args.log_path, sess.graph)

        vars = [v for v in tf.global_variables() if v.name.startswith("model/NeuCF")]
        saver = tf.train.Saver(vars, max_to_keep=200)
        for i in range(args.epochs):
            epoch_loss = 0
            for j in range(args.epoch_iter):
                batch_loss = model_train.step(sess, isTraining=True)
                epoch_loss += batch_loss / args.epoch_iter
            valid_loss = 0
            valid_sse = 0
            for j in range(args.valid_iter):
                batch_loss, batch_sse = model_valid.step(sess, isValidating=True)
                valid_loss += batch_loss / args.valid_iter
                valid_sse += batch_sse
            valid_sse = valid_sse / (args.valid_iter * args.batch_size)
            valid_rmse = np.sqrt(valid_sse)
            logfile.write( '--Avg. Train Loss ='+str(epoch_loss)[:6] + '    --Avg. Valid Loss ='+str(valid_loss)[:6]+ '    --Valid RMSE = '+str(valid_rmse)[:6]+'\n' )
            logfile.flush()
            saver.save(sess, os.path.join(args.log_path,'model'), global_step=i, write_meta_graph=False)

if __name__ == '__main__':
    args = parse_args()
    args.layers = eval(args.layers)
    args.reg_layers = eval(args.reg_layers)
    _train(args)
