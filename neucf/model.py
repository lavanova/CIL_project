import tensorflow as tf
import os
import numpy as np

def kaiming(shape, dtype, partition_info=None):

    return(tf.truncated_normal(shape, dtype=dtype)*tf.sqrt(2/float(shape[0])))

class NeuCF(object):

    def __init__(self, input, label, row_num, col_num, args):
        self.row_num = row_num
        self.col_num = col_num
        self.isTraining = tf.placeholder(tf.bool, name="isTrainingflag")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.learning_rate = tf.Variable(float(args.lr), trainable=False, dtype=tf.float32, name="learning_rate")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        classtensor = tf.constant([1,2,3,4,5], dtype=tf.float32)
        decay_steps = args.decay_step
        decay_rate = args.decay_rate
        self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps, decay_rate)
        row = input[:,0]
        col = input[:,1]
        with tf.variable_scope("NeuCF"):
            MF_Embedding_Row = tf.get_variable("mf_embedding_row", [row_num+1, args.num_factors], dtype=tf.float32,
                                                initializer=kaiming, 
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_mf)),
                                                trainable=True)
            MF_Embedding_Col = tf.get_variable("mf_embedding_col", [col_num+1, args.num_factors], dtype=tf.float32,
                                                initializer=kaiming, 
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_mf)),
                                                trainable=True)

            MLP_Embedding_Row = tf.get_variable("mlp_embedding_row", [row_num+1, int(args.layers[0]/2)], dtype=tf.float32,
                                                initializer=kaiming, 
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                trainable=True)

            MLP_Embedding_Col = tf.get_variable("mlp_embedding_col", [col_num+1, int(args.layers[0]/2)], dtype=tf.float32,
                                                initializer=kaiming, 
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                trainable=True)
        #MF part
        mf_row_latent = tf.nn.embedding_lookup(MF_Embedding_Row, row)
        mf_col_latent = tf.nn.embedding_lookup(MF_Embedding_Col, col)
        mf_vector = tf.multiply(mf_row_latent, mf_col_latent)

        #MLP part
        mlp_row_latent = tf.nn.embedding_lookup(MLP_Embedding_Row, row)
        mlp_col_latent = tf.nn.embedding_lookup(MLP_Embedding_Col, col)
        mlp_vector = tf.concat(values=[mlp_row_latent, mlp_col_latent], axis=1)
        with tf.variable_scope("NeuCF"):
            for idx in range(1, len(args.layers)):
                mlp_vector = tf.layers.dense(mlp_vector, args.layers[idx],
                                            #activation=tf.nn.relu,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[idx])),
                                            name="dense_layer%d" %idx)
                if args.batch_norm:
                    mlp_vector = tf.layers.batch_normalization(mlp_vector, training=self.isTraining, name="batch_normalization%d"%idx)
                mlp_vector = tf.nn.relu(mlp_vector)
                mlp_vector = tf.nn.dropout(mlp_vector, self.dropout_keep_prob)
        
        predict_vector = tf.concat(values=[mf_vector, mlp_vector], axis=1)
        if args.loss_type == "mse":
            with tf.variable_scope("NeuCF"):
                prediction = tf.layers.dense(predict_vector, 1,
                                            #kernel_initializer=tf.initializers.lecun_uniform,
                                            #bias_initializer=tf.initializers.lecun_uniform,
                                            name="prediction")
            self.prediction = tf.clip_by_value(prediction, 0.5, 5.5)
            self.loss1 = tf.reduce_mean( tf.square((self.prediction - label)) )
        elif args.loss_type == "cross_entropy":
            prediction = tf.layers.dense(predict_vector, 5, 
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[len(args.layers) - 1])),
                                         name="prediction")
            onetensor = tf.constant(1, dtype=tf.int32)
            self.loss1 = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=(tf.cast(tf.reshape(label,[-1]), dtype=tf.int32)-onetensor), logits=prediction) )
            probability = tf.nn.softmax(prediction)
            self.prediction = tf.reduce_sum( tf.multiply(probability, classtensor) , axis=1, keep_dims=True)
        self.sse = tf.reduce_sum( tf.square((self.prediction - label)) )
        #self.reg_loss = tf.add_n( tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) )
        self.reg_loss = sum( tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) )
        self.loss = self.loss1 + self.reg_loss
        self.rmse = tf.sqrt(self.loss1)
        self.loss_summary = tf.summary.scalar('loss/loss', self.loss)
        self.rmse_summary = tf.summary.scalar('loss/rmse', self.rmse)

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
        
        self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)

    def step(self, session, isTraining=False, isValidating=False, isTesting=False, dropout_keep_prob=0.5, logging=False):
        input_feed = {self.isTraining: isTraining,
                      self.dropout_keep_prob: dropout_keep_prob}
        if isTraining:
            if logging:
                output_feed = [self.updates, 
                               self.loss_summary,
                               self.rmse_summary,
                               self.loss,
                               self.rmse,
                               self.learning_rate_summary]
                outputs = session.run(output_feed, input_feed)
                return outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]
            
            else:
                output_feed = [self.updates,
                               self.loss]
                outputs = session.run(output_feed, input_feed)
                return outputs[1]
        
        elif isValidating:
            output_feed = [self.loss,
                           self.sse]
            outputs = session.run(output_feed, input_feed)
            return outputs[0], outputs[1]

        elif isTesting:
            outputs = session.run(self.prediction, input_feed)
            return outputs

class NeuCF2(object):

    def __init__(self, input, label, row_num, col_num, args, n_components=64):
        self.row_num = row_num
        self.col_num = col_num
        self.isTraining = tf.placeholder(tf.bool, name="isTrainingflag")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.learning_rate = tf.Variable(float(args.lr), trainable=False, dtype=tf.float32, name="learning_rate")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        classtensor = tf.constant([1,2,3,4,5], dtype=tf.float32)
        decay_steps = args.decay_step
        decay_rate = args.decay_rate
        self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps, decay_rate)
        row = input[:,0]
        col = input[:,1]
        #label = tf.cast(label, dtype=tf.int32)
        label = tf.reshape(label, [-1])
        with tf.variable_scope("NeuCF"):
            #Spectral_Embedding_Row = tf.Variable(tf.constant(0.0, shape=[row_num+1, n_components]), 
            #                                    trainable=args.external_embedding_trainable, name="spectral_embedding_row")
            if args.external_embedding_type == 0:
                Spectral_Embedding_Row = tf.get_variable(name="spectral_embedding_row", shape=[row_num+1, n_components],
                                                        dtype=tf.float32, 
                                                        #regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                        trainable=args.external_embedding_trainable)
                self.spectral_embedding_row_placeholder = tf.placeholder(tf.float32, [row_num+1, n_components])
                self.spectral_embedding_row_init = Spectral_Embedding_Row.assign(self.spectral_embedding_row_placeholder)


                #Spectral_Embedding_Col = tf.Variable(tf.constant(0.0, shape=[col_num+1, n_components]),
                #                                    trainable=args.external_embedding_trainable, name="spectral_embedding_col")
                Spectral_Embedding_Col = tf.get_variable(name="spectral_embedding_col", shape=[col_num+1, n_components],
                                                        dtype=tf.float32,
                                                        #regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                        trainable=args.external_embedding_trainable)
                self.spectral_embedding_col_placeholder = tf.placeholder(tf.float32, [col_num+1, n_components])
                self.spectral_embedding_col_init = Spectral_Embedding_Col.assign(self.spectral_embedding_col_placeholder)

                #LLE_Embedding_Row = tf.Variable(tf.constant(0.0, shape=[row_num+1, n_components]),
                #                                trainable=args.external_embedding_trainable, name="lle_embedding_row")
                LLE_Embedding_Row = tf.get_variable(name="lle_embedding_row", shape=[row_num+1, n_components],
                                                    dtype=tf.float32,
                                                    #regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                    trainable=args.external_embedding_trainable)
                self.lle_embedding_row_placeholder = tf.placeholder(tf.float32, [row_num+1, n_components])
                self.lle_embedding_row_init = LLE_Embedding_Row.assign(self.lle_embedding_row_placeholder)

                #LLE_Embedding_Col = tf.Variable(tf.constant(0.0, shape=[col_num+1, n_components]),
                #                                            trainable=args.external_embedding_trainable, name="lle_embedding_col")
                LLE_Embedding_Col = tf.get_variable(name="lle_embedding_col", shape=[col_num+1, n_components],
                                                    dtype=tf.float32,
                                                    #regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                    trainable=args.external_embedding_trainable)
                self.lle_embedding_col_placeholder = tf.placeholder(tf.float32, [col_num+1, n_components])
                self.lle_embedding_col_init = LLE_Embedding_Col.assign(self.lle_embedding_col_placeholder)

                #Factor_Embedding_Row = tf.Variable(tf.constant(0.0, shape=[row_num+1, n_components]),
                #                                trainable=args.external_embedding_trainable, name="factor_embedding_row")
                Factor_Embedding_Row = tf.get_variable(name="factor_embedding_row", shape=[row_num+1, n_components],
                                                    dtype=tf.float32,
                                                    #regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                    trainable=args.external_embedding_trainable)
                self.factor_embedding_row_placeholder = tf.placeholder(tf.float32, [row_num+1, n_components])
                self.factor_embedding_row_init = Factor_Embedding_Row.assign(self.factor_embedding_row_placeholder)

                #Factor_Embedding_Col = tf.Variable(tf.constant(0.0, shape=[col_num+1, n_components]),
                #                                trainable=args.external_embedding_trainable, name="factor_embedding_col")
                Factor_Embedding_Col = tf.get_variable(name="factor_embedding_col", shape=[col_num+1, n_components],
                                                    dtype=tf.float32,
                                                    #regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                    trainable=args.external_embedding_trainable)
                self.factor_embedding_col_placeholder = tf.placeholder(tf.float32, [col_num+1, n_components])
                self.factor_embedding_col_init = Factor_Embedding_Col.assign(self.factor_embedding_col_placeholder)

                #NMF_Embedding_Row = tf.Variable(tf.constant(0.0, shape=[row_num+1, n_components]),
                #                                trainable=args.external_embedding_trainable, name="nmf_embedding_row")
                NMF_Embedding_Row = tf.get_variable(name="nmf_embedding_row", shape=[row_num+1, n_components],
                                                    dtype=tf.float32,
                                                    #regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                    trainable=args.external_embedding_trainable)
                self.nmf_embedding_row_placeholder = tf.placeholder(tf.float32, [row_num+1, n_components])
                self.nmf_embedding_row_init = NMF_Embedding_Row.assign(self.nmf_embedding_row_placeholder)

                #NMF_Embedding_Col = tf.Variable(tf.constant(0.0, shape=[col_num+1, n_components]),
                #                                trainable=args.external_embedding_trainable, name="nmf_embedding_col")
                NMF_Embedding_Col = tf.get_variable(name="nmf_embedding_col", shape=[col_num+1, n_components],
                                                    dtype=tf.float32,
                                                    #regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                    trainable=args.external_embedding_trainable)
                self.nmf_embedding_col_placeholder = tf.placeholder(tf.float32, [col_num+1, n_components])
                self.nmf_embedding_col_init = NMF_Embedding_Col.assign(self.nmf_embedding_col_placeholder)

                spectral_row_latent = tf.nn.embedding_lookup(Spectral_Embedding_Row, row)
                spectral_col_latent = tf.nn.embedding_lookup(Spectral_Embedding_Col, col)

                lle_row_latent = tf.nn.embedding_lookup(LLE_Embedding_Row, row)
                lle_col_latent = tf.nn.embedding_lookup(LLE_Embedding_Col, col)

                factor_row_latent = tf.nn.embedding_lookup(Factor_Embedding_Row, row)
                factor_col_latent = tf.nn.embedding_lookup(Factor_Embedding_Col, col)

                nmf_row_latent = tf.nn.embedding_lookup(NMF_Embedding_Row, row)
                nmf_col_latent = tf.nn.embedding_lookup(NMF_Embedding_Col, col)

            elif args.external_embedding_type == 1:
                Graph_Embedding_Row = tf.get_variable(name="graph_embedding_row", shape=[row_num+1, args.graph_embedding_dim],
                                                    dtype=tf.float32,
                                                    #regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                    trainable=args.external_embedding_trainable)
                self.graph_embedding_row_placeholder = tf.placeholder(tf.float32, [row_num+1, args.graph_embedding_dim])
                self.graph_embedding_row_init = Graph_Embedding_Row.assign(self.graph_embedding_row_placeholder)

                Graph_Embedding_Col = tf.get_variable(name="graph_embedding_col", shape=[col_num+1, args.graph_embedding_dim],
                                                    dtype=tf.float32,
                                                    #regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                    trainable=args.external_embedding_trainable)
                self.graph_embedding_col_placeholder = tf.placeholder(tf.float32, [col_num+1, args.graph_embedding_dim])
                self.graph_embedding_col_init = Graph_Embedding_Col.assign(self.graph_embedding_col_placeholder)


                graph_row_latent = tf.nn.embedding_lookup(Graph_Embedding_Row, row)
                graph_col_latent = tf.nn.embedding_lookup(Graph_Embedding_Col, col)
        with tf.variable_scope("NeuCF"):
            """
            MF_Embedding_Row = tf.get_variable("mf_embedding_row", [row_num+1, args.num_factors], dtype=tf.float32,
                                                initializer=kaiming, 
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_mf)),
                                                trainable=True)
            MF_Embedding_Col = tf.get_variable("mf_embedding_col", [col_num+1, args.num_factors], dtype=tf.float32,
                                                initializer=kaiming, 
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_mf)),
                                                trainable=True)
            """
            MLP_Embedding_Row = tf.get_variable("mlp_embedding_row", [row_num+1, int(args.layers[0]/2)], dtype=tf.float32,
                                                initializer=kaiming, 
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                trainable=True)

            MLP_Embedding_Col = tf.get_variable("mlp_embedding_col", [col_num+1, int(args.layers[0]/2)], dtype=tf.float32,
                                                initializer=kaiming, 
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                trainable=True)
        #MF part
        #mf_row_latent = tf.nn.embedding_lookup(MF_Embedding_Row, row)
        #mf_col_latent = tf.nn.embedding_lookup(MF_Embedding_Col, col)
        #mf_vector = tf.multiply(mf_row_latent, mf_col_latent)

        #MLP part
        mlp_row_latent = tf.nn.embedding_lookup(MLP_Embedding_Row, row)
        mlp_col_latent = tf.nn.embedding_lookup(MLP_Embedding_Col, col)
        if args.external_embedding_type == 0:
            mlp_vector = tf.concat(values=[mlp_row_latent, mlp_col_latent,
                                        spectral_row_latent, spectral_col_latent,
                                        lle_row_latent, lle_col_latent,
                                        factor_row_latent, factor_col_latent,
                                        nmf_row_latent, nmf_col_latent], axis=1)
        elif args.external_embedding_type == 1:
            mlp_vector = tf.concat(values=[mlp_row_latent, mlp_col_latent,
                                           graph_row_latent, graph_col_latent], axis=1)
        if args.loss_type == "cross_entropy":
            with tf.variable_scope("NeuCF"):
                for idx in range(1, len(args.layers) - 1):
                    mlp_vector = tf.layers.dense(mlp_vector, args.layers[idx],
                                                #activation=tf.nn.relu,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[idx])),
                                                name="dense_layer%d" %idx)
                    if args.batch_norm:
                        mlp_vector = tf.layers.batch_normalization(mlp_vector, training=self.isTraining, name="batch_normalization%d"%idx)
                    mlp_vector = tf.nn.relu(mlp_vector)
                    mlp_vector = tf.nn.dropout(mlp_vector, self.dropout_keep_prob)
                mlp_vector = tf.layers.dense(mlp_vector, args.layers[len(args.layers) - 1],
                                            activation=tf.nn.relu,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[len(args.layers) - 1])),
                                            name="dense_layer%d" %(len(args.layers) - 1))
            #predict_vector = tf.concat(values=[mf_vector, mlp_vector], axis=1)
            with tf.variable_scope("NeuCF"):
                mlp_vector = tf.layers.dense(mlp_vector, 5,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[len(args.layers) - 1])),
                                            name="dense_layer_final")
            onetensor = tf.constant(1, dtype=tf.int32)
            self.loss1 = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=(tf.cast(label, dtype=tf.int32)-onetensor), logits=mlp_vector) )
            probability = tf.nn.softmax(mlp_vector)
            self.prediction = tf.reduce_sum( tf.multiply(probability, classtensor) , axis=1)
        elif args.loss_type == "mse":
            with tf.variable_scope("NeuCF"):
                for idx in range(1, len(args.layers)):
                    mlp_vector = tf.layers.dense(mlp_vector, args.layers[idx],
                                                #activation=tf.nn.relu,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[idx])),
                                                name="dense_layer%d" %idx)
                    if args.batch_norm:
                        mlp_vector = tf.layers.batch_normalization(mlp_vector, training=self.isTraining, name="batch_normalization%d"%idx)
                    mlp_vector = tf.nn.relu(mlp_vector)
                    mlp_vector = tf.nn.dropout(mlp_vector, self.dropout_keep_prob)
                #mlp_vector = tf.layers.dense(mlp_vector, args.layers[len(args.layers) - 1],
                #                            activation=tf.nn.relu,
                #                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[len(args.layers) - 1])),
                #                            name="dense_layer%d" %(len(args.layers) - 1))
            with tf.variable_scope("NeuCF"):
                mlp_vector = tf.layers.dense(mlp_vector, 8,
                                            #activation=tf.nn.relu,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[len(args.layers) - 1])),
                                            name="dense_layer_final")
                if args.batch_norm:
                    mlp_vector = tf.layers.batch_normalization(mlp_vector, training=self.isTraining, name="batch_normalization_final")
                mlp_vector = tf.nn.relu(mlp_vector)
                mlp_vector = tf.nn.dropout(mlp_vector, self.dropout_keep_prob)
                prediction = tf.layers.dense(mlp_vector, 1, name="prediction")

            prediction = tf.reshape(prediction, [-1])
            self.prediction = tf.clip_by_value(prediction, 0.5, 5.5)
            self.loss1 = tf.reduce_mean( tf.square((self.prediction - label)) )
        elif args.loss_type == "l1":
            with tf.variable_scope("NeuCF"):
                for idx in range(1, len(args.layers)):
                    mlp_vector = tf.layers.dense(mlp_vector, args.layers[idx],
                                                #activation=tf.nn.relu,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[idx])),
                                                name="dense_layer%d" %idx)
                    if args.batch_norm:
                        mlp_vector = tf.layers.batch_normalization(mlp_vector, training=self.isTraining, name="batch_normalization%d"%idx)
                    mlp_vector = tf.nn.relu(mlp_vector)
                    mlp_vector = tf.nn.dropout(mlp_vector, self.dropout_keep_prob)
                #mlp_vector = tf.layers.dense(mlp_vector, args.layers[len(args.layers) - 1],
                #                            activation=tf.nn.relu,
                #                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[len(args.layers) - 1])),
                #                            name="dense_layer%d" %(len(args.layers) - 1))
            with tf.variable_scope("NeuCF"):
                mlp_vector = tf.layers.dense(mlp_vector, 8,
                                            #activation=tf.nn.relu,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[len(args.layers) - 1])),
                                            name="dense_layer_final")
                if args.batch_norm:
                    mlp_vector = tf.layers.batch_normalization(mlp_vector, training=self.isTraining, name="batch_normalization_final")
                mlp_vector = tf.nn.relu(mlp_vector)
                mlp_vector = tf.nn.dropout(mlp_vector, self.dropout_keep_prob)
                prediction = tf.layers.dense(mlp_vector, 1, name="prediction")
                                
            prediction = tf.reshape(prediction, [-1])
            self.prediction = tf.clip_by_value(prediction, 0.5, 5.5)
            #self.loss1 = tf.reduce_mean( tf.square((self.prediction - label)) )
            self.loss1 = tf.losses.absolute_difference(label, self.prediction)
        #self.prediction = tf.clip_by_value(prediction, 0.5, 5.5)
        #self.loss1 = tf.reduce_mean( tf.square((self.prediction - label)) )
        self.sse = tf.reduce_sum( tf.square((self.prediction - label)) )
        #self.reg_loss = tf.add_n( tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) )
        self.reg_loss = sum( tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) )
        self.loss = self.loss1 + self.reg_loss
        #self.rmse = tf.sqrt(self.loss1)
        #self.loss_summary = tf.summary.scalar('loss/loss', self.loss)
        #self.rmse_summary = tf.summary.scalar('loss/rmse', self.rmse)

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
        
        self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)
    def init_embedding(self, session, args):
        if args.external_embedding_type == 0:
            row_spectral_embedding = np.load('./data/row_spectral_embedding.npy')
            col_spectral_embedding = np.load('./data/col_spectral_embedding.npy')
            row_lle_embedding = np.load('./data/row_lle_embedding.npy')
            col_lle_embedding = np.load('./data/col_lle_embedding.npy')
            row_factor_embedding = np.load('./data/row_factor_embedding.npy')
            col_factor_embedding = np.load('./data/col_factor_embedding.npy')
            row_nmf_embedding = np.load('./data/row_nmf_embedding.npy')
            col_nmf_embedding = np.load('./data/col_nmf_embedding.npy')
            input_feed = {self.spectral_embedding_row_placeholder: row_spectral_embedding,
                        self.spectral_embedding_col_placeholder: col_spectral_embedding,
                        self.lle_embedding_row_placeholder: row_lle_embedding,
                        self.lle_embedding_col_placeholder: col_lle_embedding,
                        self.factor_embedding_row_placeholder: row_factor_embedding,
                        self.factor_embedding_col_placeholder: col_factor_embedding,
                        self.nmf_embedding_row_placeholder: row_nmf_embedding,
                        self.nmf_embedding_col_placeholder: col_nmf_embedding}
            output_feed = [self.spectral_embedding_row_init,
                        self.spectral_embedding_col_init,
                        self.lle_embedding_row_init,
                        self.lle_embedding_col_init,
                        self.factor_embedding_row_init,
                        self.factor_embedding_col_init,
                        self.nmf_embedding_row_init,
                        self.nmf_embedding_col_init]
            outputs = session.run(output_feed, input_feed)
        
        elif args.external_embedding_type == 1:
            row_spectral_embedding = np.load('./data/row_spectral_embedding.npy')
            col_spectral_embedding = np.load('./data/col_spectral_embedding.npy')
            row_lle_embedding = np.load('./data/row_lle_embedding.npy')
            col_lle_embedding = np.load('./data/col_lle_embedding.npy')
            row_factor_embedding = np.load('./data/row_factor_embedding.npy')
            col_factor_embedding = np.load('./data/col_factor_embedding.npy')
            row_nmf_embedding = np.load('./data/row_nmf_embedding.npy')
            col_nmf_embedding = np.load('./data/col_nmf_embedding.npy')
            row_normal_embedding = np.concatenate((row_spectral_embedding, row_lle_embedding, row_factor_embedding, row_nmf_embedding),axis=1)
            row_normal_embedding_10000 = row_normal_embedding[1:,:]
            col_normal_embedding = np.concatenate((col_spectral_embedding, col_lle_embedding, col_factor_embedding, col_nmf_embedding), axis=1)
            col_normal_embedding_10000 = col_normal_embedding[1:,:]

            row_graph_embedding = np.load(args.graph_embedding_row_path)
            col_graph_embedding = np.load(args.graph_embedding_col_path)
            scale_row = ( np.sum(np.square(row_normal_embedding_10000)) ) / ( np.sum(np.square(row_graph_embedding)) )
            scale_row = np.sqrt(scale_row)
            scale_col = ( np.sum(np.square(col_normal_embedding_10000)) ) / ( np.sum(np.square(col_graph_embedding)) )
            scale_col = np.sqrt(scale_col)
            row_graph_embedding = row_graph_embedding * scale_row
            col_graph_embedding = col_graph_embedding * scale_col
            #row_graph_embedding = np.load(args.graph_embedding_row_path) * args.graph_embedding_scale[0]
            #col_graph_embedding = np.load(args.graph_embedding_col_path) * args.graph_embedding_scale[1]
            append_row = np.zeros((1, args.graph_embedding_dim))
            row_graph_embedding = np.concatenate( (append_row, row_graph_embedding), axis=0 )
            col_graph_embedding = np.concatenate( (append_row, col_graph_embedding), axis=0 )

            input_feed = {self.graph_embedding_row_placeholder: row_graph_embedding,
                          self.graph_embedding_col_placeholder: col_graph_embedding}
            output_feed = [self.graph_embedding_row_init,
                           self.graph_embedding_col_init]
            outputs = session.run(output_feed, input_feed)
        return
    def step(self, session, isTraining=False, isValidating=False, isTesting=False, dropout_keep_prob=0.5, logging=False):
        input_feed = {self.isTraining: isTraining,
                      self.dropout_keep_prob: dropout_keep_prob}
        if isTraining:
            if logging:
                output_feed = [self.updates, 
                               self.loss_summary,
                               self.rmse_summary,
                               self.loss,
                               self.rmse,
                               self.learning_rate_summary]
                outputs = session.run(output_feed, input_feed)
                return outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]
            
            else:
                output_feed = [self.updates,
                               self.loss]
                outputs = session.run(output_feed, input_feed)
                return outputs[1]
        
        elif isValidating:
            output_feed = [self.loss,
                           self.sse]
            outputs = session.run(output_feed, input_feed)
            return outputs[0], outputs[1]

        elif isTesting:
            outputs = session.run(self.prediction, input_feed)
            return outputs


class NeuCF3(object):

    def __init__(self, input, label, row_num, col_num, args, n_components=64):
        self.row_num = row_num
        self.col_num = col_num
        self.isTraining = tf.placeholder(tf.bool, name="isTrainingflag")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.learning_rate = tf.Variable(float(args.lr), trainable=False, dtype=tf.float32, name="learning_rate")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        classtensor = tf.constant([1,2,3,4,5], dtype=tf.float32)
        decay_steps = args.decay_step
        decay_rate = args.decay_rate
        self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps, decay_rate)
        row = input[:,0]
        col = input[:,1]
        #label = tf.cast(label, dtype=tf.int32)
        label = tf.reshape(label, [-1])
        with tf.variable_scope("NeuCF"):
            #Spectral_Embedding_Row = tf.Variable(tf.constant(0.0, shape=[row_num+1, n_components]), 
            #                                    trainable=args.external_embedding_trainable, name="spectral_embedding_row")
            Spectral_Embedding_Row = tf.get_variable(name="spectral_embedding_row", shape=[row_num+1, n_components],
                                                     dtype=tf.float32, 
                                                     regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                     trainable=args.external_embedding_trainable)
            self.spectral_embedding_row_placeholder = tf.placeholder(tf.float32, [row_num+1, n_components])
            self.spectral_embedding_row_init = Spectral_Embedding_Row.assign(self.spectral_embedding_row_placeholder)


            #Spectral_Embedding_Col = tf.Variable(tf.constant(0.0, shape=[col_num+1, n_components]),
            #                                    trainable=args.external_embedding_trainable, name="spectral_embedding_col")
            Spectral_Embedding_Col = tf.get_variable(name="spectral_embedding_col", shape=[col_num+1, n_components],
                                                     dtype=tf.float32,
                                                     regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                     trainable=args.external_embedding_trainable)
            self.spectral_embedding_col_placeholder = tf.placeholder(tf.float32, [col_num+1, n_components])
            self.spectral_embedding_col_init = Spectral_Embedding_Col.assign(self.spectral_embedding_col_placeholder)

            #LLE_Embedding_Row = tf.Variable(tf.constant(0.0, shape=[row_num+1, n_components]),
            #                                trainable=args.external_embedding_trainable, name="lle_embedding_row")
            LLE_Embedding_Row = tf.get_variable(name="lle_embedding_row", shape=[row_num+1, n_components],
                                                dtype=tf.float32,
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                trainable=args.external_embedding_trainable)
            self.lle_embedding_row_placeholder = tf.placeholder(tf.float32, [row_num+1, n_components])
            self.lle_embedding_row_init = LLE_Embedding_Row.assign(self.lle_embedding_row_placeholder)

            #LLE_Embedding_Col = tf.Variable(tf.constant(0.0, shape=[col_num+1, n_components]),
            #                                            trainable=args.external_embedding_trainable, name="lle_embedding_col")
            LLE_Embedding_Col = tf.get_variable(name="lle_embedding_col", shape=[col_num+1, n_components],
                                                dtype=tf.float32,
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                trainable=args.external_embedding_trainable)
            self.lle_embedding_col_placeholder = tf.placeholder(tf.float32, [col_num+1, n_components])
            self.lle_embedding_col_init = LLE_Embedding_Col.assign(self.lle_embedding_col_placeholder)

            #Factor_Embedding_Row = tf.Variable(tf.constant(0.0, shape=[row_num+1, n_components]),
            #                                trainable=args.external_embedding_trainable, name="factor_embedding_row")
            Factor_Embedding_Row = tf.get_variable(name="factor_embedding_row", shape=[row_num+1, n_components],
                                                   dtype=tf.float32,
                                                   regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                   trainable=args.external_embedding_trainable)
            self.factor_embedding_row_placeholder = tf.placeholder(tf.float32, [row_num+1, n_components])
            self.factor_embedding_row_init = Factor_Embedding_Row.assign(self.factor_embedding_row_placeholder)

            #Factor_Embedding_Col = tf.Variable(tf.constant(0.0, shape=[col_num+1, n_components]),
            #                                trainable=args.external_embedding_trainable, name="factor_embedding_col")
            Factor_Embedding_Col = tf.get_variable(name="factor_embedding_col", shape=[col_num+1, n_components],
                                                   dtype=tf.float32,
                                                   regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                   trainable=args.external_embedding_trainable)
            self.factor_embedding_col_placeholder = tf.placeholder(tf.float32, [col_num+1, n_components])
            self.factor_embedding_col_init = Factor_Embedding_Col.assign(self.factor_embedding_col_placeholder)

            #NMF_Embedding_Row = tf.Variable(tf.constant(0.0, shape=[row_num+1, n_components]),
            #                                trainable=args.external_embedding_trainable, name="nmf_embedding_row")
            NMF_Embedding_Row = tf.get_variable(name="nmf_embedding_row", shape=[row_num+1, n_components],
                                                dtype=tf.float32,
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                trainable=args.external_embedding_trainable)
            self.nmf_embedding_row_placeholder = tf.placeholder(tf.float32, [row_num+1, n_components])
            self.nmf_embedding_row_init = NMF_Embedding_Row.assign(self.nmf_embedding_row_placeholder)

            #NMF_Embedding_Col = tf.Variable(tf.constant(0.0, shape=[col_num+1, n_components]),
            #                                trainable=args.external_embedding_trainable, name="nmf_embedding_col")
            NMF_Embedding_Col = tf.get_variable(name="nmf_embedding_col", shape=[col_num+1, n_components],
                                                dtype=tf.float32,
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                trainable=args.external_embedding_trainable)
            self.nmf_embedding_col_placeholder = tf.placeholder(tf.float32, [col_num+1, n_components])
            self.nmf_embedding_col_init = NMF_Embedding_Col.assign(self.nmf_embedding_col_placeholder)

        spectral_row_latent = tf.nn.embedding_lookup(Spectral_Embedding_Row, row)
        spectral_col_latent = tf.nn.embedding_lookup(Spectral_Embedding_Col, col)

        lle_row_latent = tf.nn.embedding_lookup(LLE_Embedding_Row, row)
        lle_col_latent = tf.nn.embedding_lookup(LLE_Embedding_Col, col)

        factor_row_latent = tf.nn.embedding_lookup(Factor_Embedding_Row, row)
        factor_col_latent = tf.nn.embedding_lookup(Factor_Embedding_Col, col)

        nmf_row_latent = tf.nn.embedding_lookup(NMF_Embedding_Row, row)
        nmf_col_latent = tf.nn.embedding_lookup(NMF_Embedding_Col, col)

        external_row_latent = tf.concat(values=[spectral_row_latent, lle_row_latent,
                                                factor_row_latent, nmf_row_latent], 
                                        axis=1)

        external_col_latent = tf.concat(values=[spectral_col_latent, lle_col_latent,
                                                factor_col_latent, nmf_col_latent],
                                        axis=1)

        with tf.variable_scope("NeuCF"):
            
            MF_Embedding_Row = tf.get_variable("mf_embedding_row", [row_num+1, args.num_factors], dtype=tf.float32,
                                                initializer=kaiming, 
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_mf)),
                                                trainable=True)
            MF_Embedding_Col = tf.get_variable("mf_embedding_col", [col_num+1, args.num_factors], dtype=tf.float32,
                                                initializer=kaiming, 
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_mf)),
                                                trainable=True)
            
            MLP_Embedding_Row = tf.get_variable("mlp_embedding_row", [row_num+1, int(args.layers[0]/2)], dtype=tf.float32,
                                                initializer=kaiming, 
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                trainable=True)

            MLP_Embedding_Col = tf.get_variable("mlp_embedding_col", [col_num+1, int(args.layers[0]/2)], dtype=tf.float32,
                                                initializer=kaiming, 
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                trainable=True)
        #MF part
        mf_row_latent = tf.nn.embedding_lookup(MF_Embedding_Row, row)
        mf_col_latent = tf.nn.embedding_lookup(MF_Embedding_Col, col)
        #mf_row_latent = tf.concat(values=[mf_row_latent_int, external_row_latent],
        #                          axis=1)
        #mf_col_latent = tf.concat(values=[mf_col_latent_int, ])
        mf_vector = tf.multiply(mf_row_latent, mf_col_latent)

        #MLP part
        mlp_row_latent = tf.nn.embedding_lookup(MLP_Embedding_Row, row)
        mlp_col_latent = tf.nn.embedding_lookup(MLP_Embedding_Col, col)
        #mlp_vector = tf.concat(values=[mlp_row_latent, mlp_col_latent,
        #                               spectral_row_latent, spectral_col_latent,
        #                               lle_row_latent, lle_col_latent,
        #                               factor_row_latent, factor_col_latent,
        #                               nmf_row_latent, nmf_col_latent], axis=1)
        mlp_vector = tf.concat(values=[mlp_row_latent, mlp_col_latent,
                                       external_row_latent, external_col_latent],
                               axis=1)
        if args.loss_type == "cross_entropy":
            with tf.variable_scope("NeuCF"):
                """
                for idx in range(1, len(args.layers) - 1):
                    mlp_vector = tf.layers.dense(mlp_vector, args.layers[idx],
                                                #activation=tf.nn.relu,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[idx])),
                                                name="dense_layer%d" %idx)
                    if args.batch_norm:
                        mlp_vector = tf.layers.batch_normalization(mlp_vector, training=self.isTraining, name="batch_normalization%d"%idx)
                    mlp_vector = tf.nn.relu(mlp_vector)
                    mlp_vector = tf.nn.dropout(mlp_vector, self.dropout_keep_prob)
                mlp_vector = tf.layers.dense(mlp_vector, args.layers[len(args.layers) - 1],
                                            activation=tf.nn.relu,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[len(args.layers) - 1])),
                                            name="dense_layer%d" %(len(args.layers) - 1))
                """
                for idx in range(1, len(args.layers)):
                    mlp_vector = tf.layers.dense(mlp_vector, args.layers[idx],
                                                #activation=tf.nn.relu,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[idx])),
                                                name="dense_layer%d" %idx)
                    if args.batch_norm:
                        mlp_vector = tf.layers.batch_normalization(mlp_vector, training=self.isTraining, name="batch_normalization%d"%idx)
                    mlp_vector = tf.nn.relu(mlp_vector)
                    mlp_vector = tf.nn.dropout(mlp_vector, self.dropout_keep_prob)
            predict_vector = tf.concat(values=[mf_vector, mlp_vector], axis=1)
            with tf.variable_scope("NeuCF"):
                predict_vector = tf.layers.dense(predict_vector, 128,
                                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[len(args.layers) - 1])),
                                                 name="dense_layer_sub")
                if args.batch_norm:
                    predict_vector = tf.layers.batch_normalization(predict_vector, training=self.isTraining, name="batch_normalization_sub")
                predict_vector = tf.nn.relu(predict_vector)
                predict_vector = tf.nn.dropout(predict_vector, self.dropout_keep_prob)
                predict_vector = tf.layers.dense(predict_vector, 5,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[len(args.layers) - 1])),
                                            name="dense_layer_final")
            onetensor = tf.constant(1, dtype=tf.int32)
            self.loss1 = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=(tf.cast(label, dtype=tf.int32)-onetensor), logits=predict_vector) )
            probability = tf.nn.softmax(predict_vector)
            self.prediction = tf.reduce_sum( tf.multiply(probability, classtensor) , axis=1)
        elif args.loss_type == "mse":
            with tf.variable_scope("NeuCF"):
                for idx in range(1, len(args.layers)):
                    mlp_vector = tf.layers.dense(mlp_vector, args.layers[idx],
                                                #activation=tf.nn.relu,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[idx])),
                                                name="dense_layer%d" %idx)
                    if args.batch_norm:
                        mlp_vector = tf.layers.batch_normalization(mlp_vector, training=self.isTraining, name="batch_normalization%d"%idx)
                    mlp_vector = tf.nn.relu(mlp_vector)
                    mlp_vector = tf.nn.dropout(mlp_vector, self.dropout_keep_prob)
                #mlp_vector = tf.layers.dense(mlp_vector, args.layers[len(args.layers) - 1],
                #                            activation=tf.nn.relu,
                #                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[len(args.layers) - 1])),
                #                            name="dense_layer%d" %(len(args.layers) - 1))
            with tf.variable_scope("NeuCF"):
                mlp_vector = tf.layers.dense(mlp_vector, 8,
                                            #activation=tf.nn.relu,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[len(args.layers) - 1])),
                                            name="dense_layer_final")
                if args.batch_norm:
                    mlp_vector = tf.layers.batch_normalization(mlp_vector, training=self.isTraining, name="batch_normalization_final")
                mlp_vector = tf.nn.relu(mlp_vector)
                mlp_vector = tf.nn.dropout(mlp_vector, self.dropout_keep_prob)
                prediction = tf.layers.dense(mlp_vector, 1, name="prediction")

            prediction = tf.reshape(prediction, [-1])
            self.prediction = tf.clip_by_value(prediction, 0.5, 5.5)
            self.loss1 = tf.reduce_mean( tf.square((self.prediction - label)) )
        elif args.loss_type == "l1":
            with tf.variable_scope("NeuCF"):
                for idx in range(1, len(args.layers)):
                    mlp_vector = tf.layers.dense(mlp_vector, args.layers[idx],
                                                #activation=tf.nn.relu,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[idx])),
                                                name="dense_layer%d" %idx)
                    if args.batch_norm:
                        mlp_vector = tf.layers.batch_normalization(mlp_vector, training=self.isTraining, name="batch_normalization%d"%idx)
                    mlp_vector = tf.nn.relu(mlp_vector)
                    mlp_vector = tf.nn.dropout(mlp_vector, self.dropout_keep_prob)
                #mlp_vector = tf.layers.dense(mlp_vector, args.layers[len(args.layers) - 1],
                #                            activation=tf.nn.relu,
                #                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[len(args.layers) - 1])),
                #                            name="dense_layer%d" %(len(args.layers) - 1))
            with tf.variable_scope("NeuCF"):
                mlp_vector = tf.layers.dense(mlp_vector, 8,
                                            #activation=tf.nn.relu,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[len(args.layers) - 1])),
                                            name="dense_layer_final")
                if args.batch_norm:
                    mlp_vector = tf.layers.batch_normalization(mlp_vector, training=self.isTraining, name="batch_normalization_final")
                mlp_vector = tf.nn.relu(mlp_vector)
                mlp_vector = tf.nn.dropout(mlp_vector, self.dropout_keep_prob)
                prediction = tf.layers.dense(mlp_vector, 1, name="prediction")
                                
            prediction = tf.reshape(prediction, [-1])
            self.prediction = tf.clip_by_value(prediction, 0.5, 5.5)
            #self.loss1 = tf.reduce_mean( tf.square((self.prediction - label)) )
            self.loss1 = tf.losses.absolute_difference(label, self.prediction)
        #self.prediction = tf.clip_by_value(prediction, 0.5, 5.5)
        #self.loss1 = tf.reduce_mean( tf.square((self.prediction - label)) )
        self.sse = tf.reduce_sum( tf.square((self.prediction - label)) )
        #self.reg_loss = tf.add_n( tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) )
        self.reg_loss = sum( tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) )
        self.loss = self.loss1 + self.reg_loss
        #self.rmse = tf.sqrt(self.loss1)
        #self.loss_summary = tf.summary.scalar('loss/loss', self.loss)
        #self.rmse_summary = tf.summary.scalar('loss/rmse', self.rmse)

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
        
        self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)
    def init_embedding(self, session):
        row_spectral_embedding = np.load('./data/row_spectral_embedding.npy')
        col_spectral_embedding = np.load('./data/col_spectral_embedding.npy')
        row_lle_embedding = np.load('./data/row_lle_embedding.npy')
        col_lle_embedding = np.load('./data/col_lle_embedding.npy')
        row_factor_embedding = np.load('./data/row_factor_embedding.npy')
        col_factor_embedding = np.load('./data/col_factor_embedding.npy')
        row_nmf_embedding = np.load('./data/row_nmf_embedding.npy')
        col_nmf_embedding = np.load('./data/col_nmf_embedding.npy')
        input_feed = {self.spectral_embedding_row_placeholder: row_spectral_embedding,
                      self.spectral_embedding_col_placeholder: col_spectral_embedding,
                      self.lle_embedding_row_placeholder: row_lle_embedding,
                      self.lle_embedding_col_placeholder: col_lle_embedding,
                      self.factor_embedding_row_placeholder: row_factor_embedding,
                      self.factor_embedding_col_placeholder: col_factor_embedding,
                      self.nmf_embedding_row_placeholder: row_nmf_embedding,
                      self.nmf_embedding_col_placeholder: col_nmf_embedding}
        output_feed = [self.spectral_embedding_row_init,
                       self.spectral_embedding_col_init,
                       self.lle_embedding_row_init,
                       self.lle_embedding_col_init,
                       self.factor_embedding_row_init,
                       self.factor_embedding_col_init,
                       self.nmf_embedding_row_init,
                       self.nmf_embedding_col_init]
        outputs = session.run(output_feed, input_feed)
        return
    def step(self, session, isTraining=False, isValidating=False, isTesting=False, dropout_keep_prob=0.5, logging=False):
        input_feed = {self.isTraining: isTraining,
                      self.dropout_keep_prob: dropout_keep_prob}
        if isTraining:
            if logging:
                output_feed = [self.updates, 
                               self.loss_summary,
                               self.rmse_summary,
                               self.loss,
                               self.rmse,
                               self.learning_rate_summary]
                outputs = session.run(output_feed, input_feed)
                return outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]
            
            else:
                output_feed = [self.updates,
                               self.loss]
                outputs = session.run(output_feed, input_feed)
                return outputs[1]
        
        elif isValidating:
            output_feed = [self.loss,
                           self.sse]
            outputs = session.run(output_feed, input_feed)
            return outputs[0], outputs[1]

        elif isTesting:
            outputs = session.run(self.prediction, input_feed)
            return outputs   





class NeuCF4(object):

    def __init__(self, input, label, row_num, col_num, args, n_components=64):
        self.row_num = row_num
        self.col_num = col_num
        self.isTraining = tf.placeholder(tf.bool, name="isTrainingflag")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.learning_rate = tf.Variable(float(args.lr), trainable=False, dtype=tf.float32, name="learning_rate")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        classtensor = tf.constant([1,2,3,4,5], dtype=tf.float32)
        decay_steps = args.decay_step
        decay_rate = args.decay_rate
        self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps, decay_rate)
        row = input[:,0]
        col = input[:,1]
        #label = tf.cast(label, dtype=tf.int32)
        label = tf.reshape(label, [-1])
        with tf.variable_scope("NeuCF"):
            #Spectral_Embedding_Row = tf.Variable(tf.constant(0.0, shape=[row_num+1, n_components]), 
            #                                    trainable=args.external_embedding_trainable, name="spectral_embedding_row")
            Spectral_Embedding_Row = tf.get_variable(name="spectral_embedding_row", shape=[row_num+1, n_components],
                                                     dtype=tf.float32, 
                                                     regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                     trainable=args.external_embedding_trainable)
            self.spectral_embedding_row_placeholder = tf.placeholder(tf.float32, [row_num+1, n_components])
            self.spectral_embedding_row_init = Spectral_Embedding_Row.assign(self.spectral_embedding_row_placeholder)


            #Spectral_Embedding_Col = tf.Variable(tf.constant(0.0, shape=[col_num+1, n_components]),
            #                                    trainable=args.external_embedding_trainable, name="spectral_embedding_col")
            Spectral_Embedding_Col = tf.get_variable(name="spectral_embedding_col", shape=[col_num+1, n_components],
                                                     dtype=tf.float32,
                                                     regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                     trainable=args.external_embedding_trainable)
            self.spectral_embedding_col_placeholder = tf.placeholder(tf.float32, [col_num+1, n_components])
            self.spectral_embedding_col_init = Spectral_Embedding_Col.assign(self.spectral_embedding_col_placeholder)

            #LLE_Embedding_Row = tf.Variable(tf.constant(0.0, shape=[row_num+1, n_components]),
            #                                trainable=args.external_embedding_trainable, name="lle_embedding_row")
            LLE_Embedding_Row = tf.get_variable(name="lle_embedding_row", shape=[row_num+1, n_components],
                                                dtype=tf.float32,
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                trainable=args.external_embedding_trainable)
            self.lle_embedding_row_placeholder = tf.placeholder(tf.float32, [row_num+1, n_components])
            self.lle_embedding_row_init = LLE_Embedding_Row.assign(self.lle_embedding_row_placeholder)

            #LLE_Embedding_Col = tf.Variable(tf.constant(0.0, shape=[col_num+1, n_components]),
            #                                            trainable=args.external_embedding_trainable, name="lle_embedding_col")
            LLE_Embedding_Col = tf.get_variable(name="lle_embedding_col", shape=[col_num+1, n_components],
                                                dtype=tf.float32,
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                trainable=args.external_embedding_trainable)
            self.lle_embedding_col_placeholder = tf.placeholder(tf.float32, [col_num+1, n_components])
            self.lle_embedding_col_init = LLE_Embedding_Col.assign(self.lle_embedding_col_placeholder)

            #Factor_Embedding_Row = tf.Variable(tf.constant(0.0, shape=[row_num+1, n_components]),
            #                                trainable=args.external_embedding_trainable, name="factor_embedding_row")
            Factor_Embedding_Row = tf.get_variable(name="factor_embedding_row", shape=[row_num+1, n_components],
                                                   dtype=tf.float32,
                                                   regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                   trainable=args.external_embedding_trainable)
            self.factor_embedding_row_placeholder = tf.placeholder(tf.float32, [row_num+1, n_components])
            self.factor_embedding_row_init = Factor_Embedding_Row.assign(self.factor_embedding_row_placeholder)

            #Factor_Embedding_Col = tf.Variable(tf.constant(0.0, shape=[col_num+1, n_components]),
            #                                trainable=args.external_embedding_trainable, name="factor_embedding_col")
            Factor_Embedding_Col = tf.get_variable(name="factor_embedding_col", shape=[col_num+1, n_components],
                                                   dtype=tf.float32,
                                                   regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                   trainable=args.external_embedding_trainable)
            self.factor_embedding_col_placeholder = tf.placeholder(tf.float32, [col_num+1, n_components])
            self.factor_embedding_col_init = Factor_Embedding_Col.assign(self.factor_embedding_col_placeholder)

            #NMF_Embedding_Row = tf.Variable(tf.constant(0.0, shape=[row_num+1, n_components]),
            #                                trainable=args.external_embedding_trainable, name="nmf_embedding_row")
            NMF_Embedding_Row = tf.get_variable(name="nmf_embedding_row", shape=[row_num+1, n_components],
                                                dtype=tf.float32,
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                trainable=args.external_embedding_trainable)
            self.nmf_embedding_row_placeholder = tf.placeholder(tf.float32, [row_num+1, n_components])
            self.nmf_embedding_row_init = NMF_Embedding_Row.assign(self.nmf_embedding_row_placeholder)

            #NMF_Embedding_Col = tf.Variable(tf.constant(0.0, shape=[col_num+1, n_components]),
            #                                trainable=args.external_embedding_trainable, name="nmf_embedding_col")
            NMF_Embedding_Col = tf.get_variable(name="nmf_embedding_col", shape=[col_num+1, n_components],
                                                dtype=tf.float32,
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                trainable=args.external_embedding_trainable)
            self.nmf_embedding_col_placeholder = tf.placeholder(tf.float32, [col_num+1, n_components])
            self.nmf_embedding_col_init = NMF_Embedding_Col.assign(self.nmf_embedding_col_placeholder)

        spectral_row_latent = tf.nn.embedding_lookup(Spectral_Embedding_Row, row)
        spectral_col_latent = tf.nn.embedding_lookup(Spectral_Embedding_Col, col)

        lle_row_latent = tf.nn.embedding_lookup(LLE_Embedding_Row, row)
        lle_col_latent = tf.nn.embedding_lookup(LLE_Embedding_Col, col)

        factor_row_latent = tf.nn.embedding_lookup(Factor_Embedding_Row, row)
        factor_col_latent = tf.nn.embedding_lookup(Factor_Embedding_Col, col)

        nmf_row_latent = tf.nn.embedding_lookup(NMF_Embedding_Row, row)
        nmf_col_latent = tf.nn.embedding_lookup(NMF_Embedding_Col, col)

        with tf.variable_scope("NeuCF"):
            """
            MF_Embedding_Row = tf.get_variable("mf_embedding_row", [row_num+1, args.num_factors], dtype=tf.float32,
                                                initializer=kaiming, 
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_mf)),
                                                trainable=True)
            MF_Embedding_Col = tf.get_variable("mf_embedding_col", [col_num+1, args.num_factors], dtype=tf.float32,
                                                initializer=kaiming, 
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_mf)),
                                                trainable=True)
            """
            MLP_Embedding_Row = tf.get_variable("mlp_embedding_row", [row_num+1, int(args.layers[0]/2)], dtype=tf.float32,
                                                initializer=kaiming, 
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                trainable=True)

            MLP_Embedding_Col = tf.get_variable("mlp_embedding_col", [col_num+1, int(args.layers[0]/2)], dtype=tf.float32,
                                                initializer=kaiming, 
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[0])),
                                                trainable=True)
        #MF part
        #mf_row_latent = tf.nn.embedding_lookup(MF_Embedding_Row, row)
        #mf_col_latent = tf.nn.embedding_lookup(MF_Embedding_Col, col)
        #mf_vector = tf.multiply(mf_row_latent, mf_col_latent)

        #MLP part
        mlp_row_latent = tf.nn.embedding_lookup(MLP_Embedding_Row, row)
        mlp_col_latent = tf.nn.embedding_lookup(MLP_Embedding_Col, col)
        mlp_vector = tf.concat(values=[mlp_row_latent, mlp_col_latent,
                                       spectral_row_latent, spectral_col_latent,
                                       lle_row_latent, lle_col_latent,
                                       factor_row_latent, factor_col_latent,
                                       nmf_row_latent, nmf_col_latent], axis=1)
        if args.loss_type == "cross_entropy":
            with tf.variable_scope("NeuCF"):
                mlp_vector = tf.layers.dense(mlp_vector, args.layers[1],
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[1])),
                                             name="dense_layer1")
                if args.batch_norm:
                    mlp_vector = tf.layers.batch_normalization(mlp_vector, training=self.isTraining, name="batch_normalization1")
                mlp_vector = tf.nn.relu(mlp_vector)
                mlp_vector = tf.nn.dropout(mlp_vector, self.dropout_keep_prob)

                for block in range(args.residual_block):
                    mlp_vector = self.two_linear( mlp_vector, args.layers[1], block, args )

                for idx in range(2, len(args.layers) - 1):
                    mlp_vector = tf.layers.dense(mlp_vector, args.layers[idx],
                                                #activation=tf.nn.relu,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[idx])),
                                                name="dense_layer%d" %idx)
                    if args.batch_norm:
                        mlp_vector = tf.layers.batch_normalization(mlp_vector, training=self.isTraining, name="batch_normalization%d"%idx)
                    mlp_vector = tf.nn.relu(mlp_vector)
                    mlp_vector = tf.nn.dropout(mlp_vector, self.dropout_keep_prob)
                mlp_vector = tf.layers.dense(mlp_vector, args.layers[len(args.layers) - 1],
                                            activation=tf.nn.relu,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[len(args.layers) - 1])),
                                            name="dense_layer%d" %(len(args.layers) - 1))
            #predict_vector = tf.concat(values=[mf_vector, mlp_vector], axis=1)
            with tf.variable_scope("NeuCF"):
                mlp_vector = tf.layers.dense(mlp_vector, 5,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[len(args.layers) - 1])),
                                            name="dense_layer_final")
            onetensor = tf.constant(1, dtype=tf.int32)
            self.loss1 = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=(tf.cast(label, dtype=tf.int32)-onetensor), logits=mlp_vector) )
            probability = tf.nn.softmax(mlp_vector)
            self.prediction = tf.reduce_sum( tf.multiply(probability, classtensor) , axis=1)
        elif args.loss_type == "mse":
            with tf.variable_scope("NeuCF"):
                for idx in range(1, len(args.layers)):
                    mlp_vector = tf.layers.dense(mlp_vector, args.layers[idx],
                                                #activation=tf.nn.relu,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[idx])),
                                                name="dense_layer%d" %idx)
                    if args.batch_norm:
                        mlp_vector = tf.layers.batch_normalization(mlp_vector, training=self.isTraining, name="batch_normalization%d"%idx)
                    mlp_vector = tf.nn.relu(mlp_vector)
                    mlp_vector = tf.nn.dropout(mlp_vector, self.dropout_keep_prob)
                #mlp_vector = tf.layers.dense(mlp_vector, args.layers[len(args.layers) - 1],
                #                            activation=tf.nn.relu,
                #                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[len(args.layers) - 1])),
                #                            name="dense_layer%d" %(len(args.layers) - 1))
            with tf.variable_scope("NeuCF"):
                mlp_vector = tf.layers.dense(mlp_vector, 8,
                                            #activation=tf.nn.relu,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[len(args.layers) - 1])),
                                            name="dense_layer_final")
                if args.batch_norm:
                    mlp_vector = tf.layers.batch_normalization(mlp_vector, training=self.isTraining, name="batch_normalization_final")
                mlp_vector = tf.nn.relu(mlp_vector)
                mlp_vector = tf.nn.dropout(mlp_vector, self.dropout_keep_prob)
                prediction = tf.layers.dense(mlp_vector, 1, name="prediction")

            prediction = tf.reshape(prediction, [-1])
            self.prediction = tf.clip_by_value(prediction, 0.5, 5.5)
            self.loss1 = tf.reduce_mean( tf.square((self.prediction - label)) )
        elif args.loss_type == "l1":
            with tf.variable_scope("NeuCF"):
                for idx in range(1, len(args.layers)):
                    mlp_vector = tf.layers.dense(mlp_vector, args.layers[idx],
                                                #activation=tf.nn.relu,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[idx])),
                                                name="dense_layer%d" %idx)
                    if args.batch_norm:
                        mlp_vector = tf.layers.batch_normalization(mlp_vector, training=self.isTraining, name="batch_normalization%d"%idx)
                    mlp_vector = tf.nn.relu(mlp_vector)
                    mlp_vector = tf.nn.dropout(mlp_vector, self.dropout_keep_prob)
                #mlp_vector = tf.layers.dense(mlp_vector, args.layers[len(args.layers) - 1],
                #                            activation=tf.nn.relu,
                #                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[len(args.layers) - 1])),
                #                            name="dense_layer%d" %(len(args.layers) - 1))
            with tf.variable_scope("NeuCF"):
                mlp_vector = tf.layers.dense(mlp_vector, 8,
                                            #activation=tf.nn.relu,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[len(args.layers) - 1])),
                                            name="dense_layer_final")
                if args.batch_norm:
                    mlp_vector = tf.layers.batch_normalization(mlp_vector, training=self.isTraining, name="batch_normalization_final")
                mlp_vector = tf.nn.relu(mlp_vector)
                mlp_vector = tf.nn.dropout(mlp_vector, self.dropout_keep_prob)
                prediction = tf.layers.dense(mlp_vector, 1, name="prediction")
                                
            prediction = tf.reshape(prediction, [-1])
            self.prediction = tf.clip_by_value(prediction, 0.5, 5.5)
            #self.loss1 = tf.reduce_mean( tf.square((self.prediction - label)) )
            self.loss1 = tf.losses.absolute_difference(label, self.prediction)
        #self.prediction = tf.clip_by_value(prediction, 0.5, 5.5)
        #self.loss1 = tf.reduce_mean( tf.square((self.prediction - label)) )
        self.sse = tf.reduce_sum( tf.square((self.prediction - label)) )
        #self.reg_loss = tf.add_n( tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) )
        self.reg_loss = sum( tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) )
        self.loss = self.loss1 + self.reg_loss
        #self.rmse = tf.sqrt(self.loss1)
        #self.loss_summary = tf.summary.scalar('loss/loss', self.loss)
        #self.rmse_summary = tf.summary.scalar('loss/rmse', self.rmse)

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
        
        self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)
    
    def two_linear(self, xin, linear_size, idx, args):
        #regular scale!!!
        
        with tf.variable_scope("two_linear_" + str(idx)):
            input_size = int(xin.get_shape()[1])
            w2 = tf.get_variable(name='w2_'+str(idx), initializer=kaiming, 
                                 shape=[input_size, linear_size], dtype=tf.float32)
            b2 = tf.get_variable(name='b2_'+str(idx), initializer=kaiming,
                                 shape=[linear_size], dtype=tf.float32)
            w2 = tf.clip_by_norm(w2, 1) if args.max_norm_res else w2
            y = tf.matmul(xin, w2) + b2
            #y = tf.layers.dense(xin, linear_size, 
            #                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[1])),
            #                    name="interdense1_"+str(idx))
            if args.batch_norm:
                y = tf.layers.batch_normalization(y, training=self.isTraining, name="batch_normalization1_"+str(idx))
            y = tf.nn.relu(y)
            y = tf.nn.dropout(y, self.dropout_keep_prob)

            #y = tf.layers.dense(y, linear_size, 
            #                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[1])),
            #                    name="interdense2_"+str(idx))
            w3 = tf.get_variable(name='w3_'+str(idx), initializer=kaiming, 
                                 shape=[linear_size, linear_size], dtype=tf.float32)
            b3 = tf.get_variable(name='b3_'+str(idx), initializer=kaiming,
                                 shape=[linear_size], dtype=tf.float32)
            w3 = tf.clip_by_norm(w3, 1) if args.max_norm_res else w3
            y = tf.matmul(y, w3) + b3
            if args.batch_norm:
                y = tf.layers.batch_normalization(y, training=self.isTraining, name="batch_normalization2_"+str(idx))
            y = tf.nn.relu(y)
            y = tf.nn.dropout(y, self.dropout_keep_prob)

            y = (xin + y) if args.residual else y
        
        return y
            

    def init_embedding(self, session):
        row_spectral_embedding = np.load('./data/row_spectral_embedding.npy')
        col_spectral_embedding = np.load('./data/col_spectral_embedding.npy')
        row_lle_embedding = np.load('./data/row_lle_embedding.npy')
        col_lle_embedding = np.load('./data/col_lle_embedding.npy')
        row_factor_embedding = np.load('./data/row_factor_embedding.npy')
        col_factor_embedding = np.load('./data/col_factor_embedding.npy')
        row_nmf_embedding = np.load('./data/row_nmf_embedding.npy')
        col_nmf_embedding = np.load('./data/col_nmf_embedding.npy')
        input_feed = {self.spectral_embedding_row_placeholder: row_spectral_embedding,
                      self.spectral_embedding_col_placeholder: col_spectral_embedding,
                      self.lle_embedding_row_placeholder: row_lle_embedding,
                      self.lle_embedding_col_placeholder: col_lle_embedding,
                      self.factor_embedding_row_placeholder: row_factor_embedding,
                      self.factor_embedding_col_placeholder: col_factor_embedding,
                      self.nmf_embedding_row_placeholder: row_nmf_embedding,
                      self.nmf_embedding_col_placeholder: col_nmf_embedding}
        output_feed = [self.spectral_embedding_row_init,
                       self.spectral_embedding_col_init,
                       self.lle_embedding_row_init,
                       self.lle_embedding_col_init,
                       self.factor_embedding_row_init,
                       self.factor_embedding_col_init,
                       self.nmf_embedding_row_init,
                       self.nmf_embedding_col_init]
        outputs = session.run(output_feed, input_feed)
        return
    def step(self, session, isTraining=False, isValidating=False, isTesting=False, dropout_keep_prob=0.5, logging=False):
        input_feed = {self.isTraining: isTraining,
                      self.dropout_keep_prob: dropout_keep_prob}
        if isTraining:
            if logging:
                output_feed = [self.updates, 
                               self.loss_summary,
                               self.rmse_summary,
                               self.loss,
                               self.rmse,
                               self.learning_rate_summary]
                outputs = session.run(output_feed, input_feed)
                return outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]
            
            else:
                output_feed = [self.updates,
                               self.loss]
                outputs = session.run(output_feed, input_feed)
                return outputs[1]
        
        elif isValidating:
            output_feed = [self.loss,
                           self.sse]
            outputs = session.run(output_feed, input_feed)
            return outputs[0], outputs[1]

        elif isTesting:
            outputs = session.run(self.prediction, input_feed)
            return outputs








