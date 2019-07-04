import numpy as np
import tensorflow as tf
#from data import create_dataloader_train, create_dataloader_test
from ngcf_data import Data
import argparse
from model import NeuCF, NeuCF2, NeuCF3, NeuCF4
import os
import math
import pandas as pd
from utils import early_stopping
from tqdm import tqdm
import sys
from shutil import copyfile

def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--mode', type=int, default=0,
                        help='0: training; 1: inference on valid set and test set with pretrained model')
    parser.add_argument('--model_path', nargs='?', default='',
                        help='load path of pretrained model')     
    parser.add_argument('--test_path', nargs='?', default='',
                        help='when in mode 1, path of output of test')
    parser.add_argument('--output_valid_path', nargs='?', default='',
                        help='when in mode 1, path of output of output_valid') 
    parser.add_argument('--data_path', nargs='?', default='../data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')
    parser.add_argument('--log_path', nargs='?', default='./log/model/',
                        help='log path')
    parser.add_argument('--flag_step', type=int, default=20,
                        help='flag step')
    parser.add_argument('--dataset', nargs='?', default='gowalla',
                        help='Choose a dataset from {gowalla, yelp2018, amazon-book}')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epoch.')
    parser.add_argument('--epoch_iter', type=int, default=1150,
                        help='how many batches in one epoch')
    parser.add_argument('--dense_layer_type', type=int, default=0,
                        help='dense layer type')
    parser.add_argument('--dense_layer_size', nargs='?', default='[256,1024,512,256,128]',
                        help='dense layer size')
    parser.add_argument('--dense_layer_regs', nargs='?', default='[0.00001,0.00001,0.00001,0.00001,0.00001]',
                        help='regularization scale for dense layer')
    parser.add_argument('--batch_norm', type=bool, default=True,
                        help='whether use batch normalization')
    parser.add_argument('--valid_iter', type=int, default=40,
                        help='how many batches used to do validation')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='train valid split ratio')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--end_to_end', type=bool, default=False,
                        help='whether add dense layers')
    parser.add_argument('--model_type', nargs='?', default='ngcf',
                        help='Specify the name of model (ngcf).')
    parser.add_argument('--adj_type', nargs='?', default='norm',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='ngcf',
                        help='Specify the type of the graph convolutional layer from {ngcf, gcn, gcmc}.')
    parser.add_argument('--loss_type', nargs='?', default='mse',
                        help='mse, l1, cross_entropy')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')
    parser.add_argument('--dense_layer_dropout_keep_prob', type=float, default=0.5,
                        help='dropout_keep_prob for dense layer')
    parser.add_argument('--node_dropout_flag', type=int, default=0,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    return parser.parse_args()
def kaiming(shape, dtype, partition_info=None):

    return(tf.truncated_normal(shape, dtype=dtype)*tf.sqrt(2/float(shape[0])))
class NGCF(object):
    def __init__(self, row_col, label, data_config, args, pretrain_data=None):
        # argument settings
        self.model_type = 'ngcf'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 100

        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = args.layer_size
        self.n_layers = len(self.weight_size)

        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.regs = args.regs
        self.decay = self.regs[0]

        self.verbose = args.verbose

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        #self.users = tf.placeholder(tf.int32, shape=(None,))
        row = row_col[:, 0]
        #self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        col = row_col[:,1]
        #self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        #self.pos_items_rating = tf.placeholder(tf.int32, shape=(None,))

        # dropout: node dropout (adopted on the ego-networks);
        #          ... since the usage of node dropout have higher computational cost,
        #          ... please use the 'node_dropout_flag' to indicate whether use such technique.
        #          message dropout (adopted on the convolution operations).
        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])
        classtensor = tf.constant([1,2,3,4,5], dtype=tf.float32)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.isTraining = tf.placeholder(tf.bool, name="isTrainingflag")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        with tf.variable_scope("ngcf"):
            self.weights = self._init_weights()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
        """
        if self.alg_type in ['ngcf']:
            self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()

        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['gcmc']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        #self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        #self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        #self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)
        row_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, row)
        col_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, col)

        #if args.loss_type == 'mse':
        #    prediction = tf.reduce_sum(tf.multiply(row_g_embeddings, col_g_embeddings), axis=1)
        if args.end_to_end:
            if args.dense_layer_type == 0:
                with tf.variable_scope("ngcf"):
                    MLP_Embedding_Row = tf.get_variable("mlp_embedding_row", [self.n_users, int(args.dense_layer_size[0]/2)], dtype=tf.float32,
                                                        initializer=kaiming, 
                                                        regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.dense_layer_regs[0])),
                                                        trainable=True)

                    MLP_Embedding_Col = tf.get_variable("mlp_embedding_col", [self.n_items, int(args.dense_layer_size[0]/2)], dtype=tf.float32,
                                                        initializer=kaiming, 
                                                        regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.dense_layer_regs[0])),
                                                        trainable=True)      
                mlp_row_latent = tf.nn.embedding_lookup(MLP_Embedding_Row, row)
                mlp_col_latent = tf.nn.embedding_lookup(MLP_Embedding_Col, col)   
                mlp_vector = tf.concat(values=[mlp_row_latent, mlp_col_latent,
                                            row_g_embeddings, col_g_embeddings], axis=1)  
            
            elif args.dense_layer_type == 1:
                mlp_vector = tf.concat(values=[row_g_embeddings, col_g_embeddings],
                                       axis=1)
            
            if args.loss_type == "cross_entropy":
                with tf.variable_scope("ngcf"):
                    for idx in range(1, len(args.dense_layer_size) - 1):
                        mlp_vector = tf.layers.dense(mlp_vector, args.dense_layer_size[idx],
                                                    #activation=tf.nn.relu,
                                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.dense_layer_regs[idx])),
                                                    name="dense_layer%d" %idx)
                        if args.batch_norm:
                            mlp_vector = tf.layers.batch_normalization(mlp_vector, training=self.isTraining, name="batch_normalization%d"%idx)
                        mlp_vector = tf.nn.relu(mlp_vector)
                        mlp_vector = tf.nn.dropout(mlp_vector, self.dropout_keep_prob)
                    mlp_vector = tf.layers.dense(mlp_vector, args.dense_layer_size[len(args.dense_layer_size) - 1],
                                                activation=tf.nn.relu,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.dense_layer_regs[len(args.dense_layer_size) - 1])),
                                                name="dense_layer%d" %(len(args.dense_layer_size) - 1))
                #predict_vector = tf.concat(values=[mf_vector, mlp_vector], axis=1)
                with tf.variable_scope("ngcf"):
                    mlp_vector = tf.layers.dense(mlp_vector, 5,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.dense_layer_regs[len(args.dense_layer_size) - 1])),
                                                name="dense_layer_final")
                onetensor = tf.constant(1, dtype=tf.int32)
                self.mf_loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=(tf.cast(tf.reshape(label,[-1]), dtype=tf.int32)-onetensor), logits=mlp_vector) )
                probability = tf.nn.softmax(mlp_vector)
                self.prediction = tf.reduce_sum( tf.multiply(probability, classtensor) , axis=1)
        else:
            if args.loss_type == 'mse':
                row_col_g_embeddings = tf.concat([row_g_embeddings, col_g_embeddings], axis=1)
                with tf.variable_scope("ngcf"):
                    prediction = tf.layers.dense(row_col_g_embeddings, 1, 
                                                 #kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.regs[0])),
                                                 name="prediction")
                self.prediction = tf.reshape(tf.clip_by_value(prediction, 0.5, 5.5), [-1])
                self.mf_loss = tf.losses.mean_squared_error(label, self.prediction)
            elif args.loss_type == 'cross_entropy':
                row_col_g_embeddings = tf.concat([row_g_embeddings, col_g_embeddings], axis=1)
                with tf.variable_scope("ngcf"):
                    prediction = tf.layers.dense(row_col_g_embeddings, 5, 
                                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.regs[0])),
                                                 name="prediction")
                onetensor = tf.constant(1, dtype=tf.int32)
                self.mf_loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=(tf.cast(tf.reshape(label,[-1]), dtype=tf.int32)-onetensor), logits=prediction) )
                probability = tf.nn.softmax(prediction)
                self.prediction = tf.reduce_sum(tf.multiply(probability, classtensor), axis=1)
                #regularizer = tf.nn.l2_loss(row_g_embeddings) + tf.nn.l2_loss(col_g_embeddings) 
                #regularizer = regularizer / self.batch_size  
                #self.emb_loss = self.decay * regularizer   
                #self.reg_loss = tf.constant(0.0, tf.float32, [1])
                #self.loss = self.mf_loss + self.emb_loss + self.reg_loss
        
        regularizer = tf.nn.l2_loss(row_g_embeddings) + tf.nn.l2_loss(col_g_embeddings) 
        regularizer = regularizer / self.batch_size  

        self.emb_loss = self.decay * regularizer   
        #self.reg_loss = tf.constant(0.0, tf.float32, [1])
        self.reg_loss = sum( tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) )
        self.loss = self.mf_loss + self.emb_loss + self.reg_loss
        self.sse = tf.reduce_sum( tf.square((self.prediction - label)) )
        opt = tf.train.AdamOptimizer(args.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            if args.loss_type == "cross_entropy":
                gradients = opt.compute_gradients(self.loss)
                self.gradients = [[] if i == None else i for i in gradients]
                grads, variables = zip(*gradients)
                grads, _ = tf.clip_by_global_norm(grads, 5.0)
                self.updates = opt.apply_gradients(zip(grads, variables), global_step=self.global_step)
            else:            
                gradients = opt.compute_gradients(self.loss)
                self.gradients = [[] if i == None else i for i in gradients]
                self.updates = opt.apply_gradients(gradients, global_step=self.global_step)
        """
        pos_score = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        pos_score = tf.clip_by_value(pos_score, 0.5, 5.5)
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) 
        regularizer = regularizer/self.batch_size  

        mf_loss = tf.reduce_mean(tf.square(pos_score - tf.cast(pos_items_rating, dtype=tf.float32)))
        emb_loss = self.decay * regularizer   
        reg_loss = tf.constant(0.0, tf.float32, [1])
        """
        """
        *********************************************************
        Inference for the testing phase.
        """
        #self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)
        #self.batch_ratings = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.pos_i_g_embeddings), axis=1)
        #self.batch_ratings = tf.clip_by_value(self.batch_ratings, 0.5, 5)
        #self.sse = tf.(self.batch_ratings - tf.cast(self.pos_items_rating))
        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        #self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
        #                                                                  self.pos_i_g_embeddings,
        #                                                                  self.neg_i_g_embeddings)
        #self.mf_loss, self.emb_loss, self.reg_loss = self.create_new_loss(self.u_g_embeddings,
        #                                                                  self.pos_i_g_embeddings,
        #                                                                  self.pos_items_rating)
        #self.loss = self.mf_loss + self.emb_loss + self.reg_loss

        #self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
    def step(self, session, node_dropout, mess_dropout, dropout_keep_prob=0.5, isTraining=False, isValidating=False, isTesting=False, logging=False):
        #input_feed = {self.isTraining: isTraining,
        #              self.dropout_keep_prob: dropout_keep_prob}
        input_feed = {self.isTraining: isTraining,
                      self.node_dropout: node_dropout,
                      self.mess_dropout: mess_dropout,
                      self.dropout_keep_prob: dropout_keep_prob}
        if isTraining:
            if logging:
                #output_feed = [self.updates, 
                #               self.loss_summary,
                #               self.rmse_summary,
                #               self.loss,
                #               self.rmse,
                #               self.learning_rate_summary]
                #outputs = session.run(output_feed, input_feed)
                #return outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]
                pass
            else:
                output_feed = [self.updates,
                               self.loss,
                               self.mf_loss]
                outputs = session.run(output_feed, input_feed)
                return outputs[1], outputs[2]
        
        elif isValidating:
            output_feed = [self.loss,
                           self.mf_loss,
                           self.sse]
            outputs = session.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]
        
        elif isTesting:
            outputs = session.run(self.prediction, input_feed)
            return outputs   
        
    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            #all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            #all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            all_weights['user_embedding'] = tf.get_variable(name='user_embedding', shape=[self.n_users, self.emb_dim],
                                                            initializer=initializer,
                                                            trainable=True)
            all_weights['item_embedding'] = tf.get_variable(name='item_embedding', shape=[self.n_items, self.emb_dim],
                                                            initializer=initializer,
                                                            trainable=True)
            print('using xavier initialization')
        else:
            print("not implement pretrain embedding")
            """
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')
            """
        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            """
            all_weights['W_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)
            """
            all_weights['W_gc_%d' %k] = tf.get_variable(name='W_gc_%d' % k, shape=[self.weight_size_list[k], self.weight_size_list[k+1]],
                                                        initializer=initializer, trainable=True)
            all_weights['b_gc_%d' %k] = tf.get_variable(name='b_gc_%d' % k, shape=[1, self.weight_size_list[k+1]],
                                                        initializer=initializer, trainable=True)

            all_weights['W_bi_%d' % k] = tf.get_variable(name='W_bi_%d' % k, shape=[self.weight_size_list[k], self.weight_size_list[k + 1]],
                                                         initializer=initializer, trainable=True)
            all_weights['b_bi_%d' % k] = tf.get_variable(name='b_bi_%d' % k, shape=[1, self.weight_size_list[k + 1]],
                                                         initializer=initializer, trainable=True)

            all_weights['W_mlp_%d' % k] = tf.get_variable(name='W_mlp_%d' % k, shape=[self.weight_size_list[k], self.weight_size_list[k+1]],
                                                          initializer=initializer, trainable=True)
            all_weights['b_mlp_%d' % k] = tf.get_variable(name='b_mlp_%d' % k, shape=[1, self.weight_size_list[k+1]],
                                                          initializer=initializer, trainable=True)
        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_ngcf_embed(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            #norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcn_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)


        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' %k]) + self.weights['b_gc_%d' %k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcmc_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = []

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            # convolutional layer.
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # dense layer.
            mlp_embeddings = tf.matmul(embeddings, self.weights['W_mlp_%d' %k]) + self.weights['b_mlp_%d' %k]
            mlp_embeddings = tf.nn.dropout(mlp_embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [mlp_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def create_new_loss(self, users, pos_items, pos_items_rating):
        pos_score = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        pos_score = tf.clip_by_value(pos_score, 0.5, 5.5)
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) 
        regularizer = regularizer/self.batch_size  

        mf_loss = tf.reduce_mean(tf.square(pos_score - tf.cast(pos_items_rating, dtype=tf.float32)))
        emb_loss = self.decay * regularizer   
        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss
    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        mf_loss = tf.negative(tf.reduce_mean(maxi))

        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)


if __name__ == '__main__':
    args = parse_args()
    args.layer_size = eval(args.layer_size)
    args.regs = eval(args.regs)
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    args.Ks = eval(args.Ks)
    args.dense_layer_size = eval(args.dense_layer_size)
    args.dense_layer_regs = eval(args.dense_layer_regs)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    data_config = dict()
    data_config['n_users'] = 10000
    data_config['n_items'] = 1000
    os.makedirs(os.path.join(args.log_path), exist_ok=True)
    logfile = open(os.path.join(args.log_path, 'log.txt'), 'w', buffering=1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"

    with tf.Session(config=config) as sess:
        data_generator = Data(batch_size=args.batch_size, valid_ratio=args.valid_ratio)
        

        plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

        if args.adj_type == 'plain':
            data_config['norm_adj'] = plain_adj
            print('use the plain adjacency matrix')

        elif args.adj_type == 'norm':
            data_config['norm_adj'] = norm_adj
            print('use the normalized adjacency matrix')

        elif args.adj_type == 'gcmc':
            data_config['norm_adj'] = mean_adj
            print('use the gcmc adjacency matrix')

        else:
            data_config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
            print('use the mean adjacency matrix')
        with tf.variable_scope("model", reuse=False):
            model_train = NGCF(data_generator.row_col_train, data_generator.label_train, data_config, args, pretrain_data=None)
            sess.run(tf.global_variables_initializer())
        with tf.variable_scope("model", reuse=True):
            model_valid = NGCF(data_generator.row_col_valid, data_generator.label_valid, data_config, args, pretrain_data=None)
            sess.run(tf.global_variables_initializer())
        sess.run([data_generator.iterator_test.initializer,
                  data_generator.iterator_output.initializer,
                  data_generator.iterator_output_valid.initializer])
        dataloader_test = data_generator.iterator_test.get_next()
        dataloader_output = data_generator.iterator_output.get_next()
        dataloader_output_valid = data_generator.iterator_output_valid.get_next()
        row_col_test, label_test = dataloader_test
        row_col_output, label_output = dataloader_output
        row_col_output_valid, label_output_valid = dataloader_output_valid
        with tf.variable_scope("model", reuse=True):
            model_test = NGCF(row_col_test, label_test, data_config, args, pretrain_data=None)
            sess.run(tf.global_variables_initializer())
        with tf.variable_scope("model", reuse=True):
            model_output = NGCF(row_col_output, label_output, data_config, args, pretrain_data=None)
            sess.run(tf.global_variables_initializer())
        with tf.variable_scope("model", reuse=True):
            model_output_valid = NGCF(row_col_output_valid, label_output_valid, data_config, args, pretrain_data=None)
            sess.run(tf.global_variables_initializer())

        #vars = [v for v in tf.global_variables() if v.name.startswith("model/NeuCF")]
        saver = tf.train.Saver(max_to_keep=200)       

        if args.mode == 1:
            saver.restore(sess, args.model_path)
            output_valid_prediction = None
            for j in tqdm(range( math.ceil( len(data_generator.rcstrs_output_valid) / args.batch_size) )):
                predict = model_output_valid.step(sess, [0.]*len(args.layer_size), [0.]*len(args.layer_size), dropout_keep_prob=1, isTesting=True)
                if j == 0:
                    output_valid_prediction = predict
                else:
                    output_valid_prediction = np.concatenate( [output_valid_prediction, predict] , axis=0 )
            output_valid_prediction = np.reshape(output_valid_prediction, (output_valid_prediction.shape[0],))
            df = pd.DataFrame( {'Id': data_generator.rcstrs_output_valid,'Prediction': output_valid_prediction} )
            df.to_csv(args.output_valid_path, index=False)
            test_prediction = None
            for j in tqdm(range(math.ceil( len(data_generator.rcstrs_test) / args.batch_size))):
                predict = model_test.step(sess, [0.]*len(args.layer_size), [0.]*len(args.layer_size), dropout_keep_prob=1, isTesting=True)
                if j == 0:
                    test_prediction = predict
                else:
                    test_prediction = np.concatenate([test_prediction, predict], axis=0)
            test_prediction = np.reshape(test_prediction, (test_prediction.shape[0],))
            
            # data frame is reconstructed since the direct modification is too slow
            df = pd.DataFrame({'Id': data_generator.rcstrs_test,'Prediction': test_prediction})
            df.to_csv(args.test_path, index=False)            
            sys.exit()
        should_stop = False
        stopping_step = 0
        cur_best_pre_0 = 1.0
        for i in range(args.epoch):
            epoch_loss = 0
            epoch_mf_loss = 0
            for j in tqdm(range(args.epoch_iter)):
                batch_loss, batch_mf_loss = model_train.step(sess, args.node_dropout, args.mess_dropout, dropout_keep_prob=args.dense_layer_dropout_keep_prob, isTraining=True)
                epoch_loss += batch_loss / args.epoch_iter
                epoch_mf_loss += batch_mf_loss / args.epoch_iter
            valid_loss = 0
            valid_mf_loss = 0
            valid_sse = 0
            for j in tqdm(range(args.valid_iter)):
                batch_loss, batch_mf_loss, batch_sse = model_valid.step(sess, [0.]*len(args.layer_size), [0.]*len(args.layer_size), dropout_keep_prob=1, isValidating=True)
                valid_loss += batch_loss / args.valid_iter
                valid_mf_loss += batch_mf_loss / args.valid_iter
                valid_sse += batch_sse
            valid_sse = valid_sse / (args.valid_iter * args.batch_size)
            valid_rmse = np.sqrt(valid_sse)
            print('--Avg. Train Loss ='+str(epoch_loss)[:6] + '    --Avg. Train MF Loss ='+ str(epoch_mf_loss)[:6]+'    --Avg. Valid Loss ='+str(valid_loss)[:6]+ '     --Avg. Valid MF Loss ='+str(valid_mf_loss)[:6] +'    --Valid RMSE = '+str(valid_rmse)[:6])
            logfile.write( '--Avg. Train Loss ='+str(epoch_loss)[:6] + '    --Avg. Train MF Loss ='+ str(epoch_mf_loss)[:6]+'    --Avg. Valid Loss ='+str(valid_loss)[:6]+ '     --Avg. Valid MF Loss ='+str(valid_mf_loss)[:6] +'    --Valid RMSE = '+str(valid_rmse)[:6]+'\n' )
            logfile.flush()
            cur_best_pre_0, stopping_step, should_stop = early_stopping(valid_rmse, cur_best_pre_0, 
                                                                        stopping_step, expected_order='dec',
                                                                        flag_step=args.flag_step)
            if should_stop == True:
                break
            if valid_rmse == cur_best_pre_0 and args.save_flag == 1:
                saver.save(sess, os.path.join(args.log_path,'model'), global_step=i+1, write_meta_graph=False)
                
                input_feed = {model_valid.node_dropout: [0.]*len(args.layer_size),
                              model_valid.mess_dropout: [0.]*len(args.layer_size)}
                output_feed = [model_valid.ua_embeddings,
                               model_valid.ia_embeddings]
                ua_embeddings, ia_embeddings = sess.run(output_feed, 
                                                        feed_dict=input_feed)
                np.save(os.path.join(args.log_path)+'row_embedding_'+str(i+1)+'.npy', ua_embeddings)
                np.save(os.path.join(args.log_path)+'row_embedding.npy', ua_embeddings)

                np.save(os.path.join(args.log_path)+'col_embedding_'+str(i+1)+'.npy', ia_embeddings)
                np.save(os.path.join(args.log_path)+'col_embedding.npy', ia_embeddings)
                output_prediction = None
                for j in tqdm(range( math.ceil( len(data_generator.rcstrs_output) / args.batch_size) )):
                    predict = model_output.step(sess, [0.]*len(args.layer_size), [0.]*len(args.layer_size), dropout_keep_prob=1, isTesting=True)
                    if j == 0:
                        output_prediction = predict
                    else:
                        output_prediction = np.concatenate([output_prediction, predict], axis=0)
                output_prediction = np.reshape(output_prediction, (output_prediction.shape[0],))
                df = pd.DataFrame( {'Id': data_generator.rcstrs_output,'Prediction': output_prediction} )
                df.to_csv(os.path.join(args.log_path, 'output' + str(i+1)+".csv" ),index=False)

                output_valid_prediction = None
                for j in tqdm(range( math.ceil( len(data_generator.rcstrs_output_valid) / args.batch_size) )):
                    predict = model_output_valid.step(sess, [0.]*len(args.layer_size), [0.]*len(args.layer_size), dropout_keep_prob=1, isTesting=True)
                    if j == 0:
                        output_valid_prediction = predict
                    else:
                        output_valid_prediction = np.concatenate( [output_valid_prediction, predict] , axis=0 )
                output_valid_prediction = np.reshape(output_valid_prediction, (output_valid_prediction.shape[0],))
                df = pd.DataFrame( {'Id': data_generator.rcstrs_output_valid,'Prediction': output_valid_prediction} )
                df.to_csv(os.path.join(args.log_path, 'output_valid' + str(i+1)+".csv" ),index=False)
                if args.end_to_end:
                    copyfile( os.path.join(args.log_path, 'output_valid' + str(i+1)+".csv" ) , 
                              os.path.join("../cache", args.log_path.split("/")[-2] ) )
                test_prediction = None
                for j in tqdm(range(math.ceil( len(data_generator.rcstrs_test) / args.batch_size))):
                    predict = model_test.step(sess, [0.]*len(args.layer_size), [0.]*len(args.layer_size), dropout_keep_prob=1, isTesting=True)
                    if j == 0:
                        test_prediction = predict
                    else:
                        test_prediction = np.concatenate([test_prediction, predict], axis=0)
                test_prediction = np.reshape(test_prediction, (test_prediction.shape[0],))
                
                # data frame is reconstructed since the direct modification is too slow
                df = pd.DataFrame({'Id': data_generator.rcstrs_test,'Prediction': test_prediction})
                df.to_csv(os.path.join(args.log_path, 'submission' + str(i+1)+".csv" ),index=False)
                if args.end_to_end:
                    copyfile( os.path.join(args.log_path, 'submission' + str(i+1)+".csv" ) , 
                                            os.path.join("../test", args.log_path.split("/")[-2]) )
                sess.run([data_generator.iterator_test.initializer,
                          data_generator.iterator_output.initializer,
                          data_generator.iterator_output_valid.initializer])
                dataloader_test = data_generator.iterator_test.get_next()
                dataloader_output = data_generator.iterator_output.get_next()
                dataloader_output_valid = data_generator.iterator_output_valid.get_next()
                row_col_test, label_test = dataloader_test
                row_col_output, label_output = dataloader_output
                row_col_output_valid, label_output_valid = dataloader_output_valid

    #_train(args)
