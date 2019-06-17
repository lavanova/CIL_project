import tensorflow as tf
import os

def kaiming(shape, dtype, partition_info=None):

    return(tf.truncated_normal(shape, dtype=dtype)*tf.sqrt(2/float(shape[0])))

class NeuCF(object):

    def __init__(self, input, label, row_num, col_num, args):
        self.row_num = row_num
        self.col_num = col_num
        self.learning_rate = tf.Variable(float(args.lr), trainable=False, dtype=tf.float32, name="learning_rate")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        decay_steps = 100000
        decay_rate = 0.96
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
                                            activation=tf.nn.relu,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=float(args.reg_layers[idx])),
                                            name="dense_layer%d" %idx)
        
        predict_vector = tf.concat(values=[mf_vector, mlp_vector], axis=1)
        with tf.variable_scope("NeuCF"):
            self.prediction = tf.layers.dense(predict_vector, 1,
                                        #kernel_initializer=tf.initializers.lecun_uniform,
                                        #bias_initializer=tf.initializers.lecun_uniform,
                                        name="prediction")
        
        self.loss1 = tf.reduce_mean( tf.square((self.prediction - label)) )
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
            gradients = opt.compute_gradients(self.loss)
            self.gradients = [[] if i == None else i for i in gradients]
            self.updates = opt.apply_gradients(gradients, global_step=self.global_step)
        
        self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)

    def step(self, session, isTraining=False, isValidating=False, isTesting=False, logging=False):
        if isTraining:
            if logging:
                output_feed = [self.updates, 
                               self.loss_summary,
                               self.rmse_summary,
                               self.loss,
                               self.rmse,
                               self.learning_rate_summary]
                outputs = session.run(output_feed)
                return outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]
            
            else:
                output_feed = [self.updates,
                               self.loss]
                outputs = session.run(output_feed)
                return outputs[1]
        
        elif isValidating:
            output_feed = [self.loss_summary,
                           self.rmse_summary,
                           self.loss,
                           self.sse]
            outputs = session.run(output_feed)
            return outputs[2], outputs[3]

        elif isTesting:
            outputs = session.run(self.prediction)
            return outputs
