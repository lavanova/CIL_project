import numpy as np
import tensorflow as tf
import os
from utils import *
from dataset import _get_training_data, _get_test_data
from train_model import TrainModel
from sklearn.metrics import mean_absolute_error, mean_squared_error


tf.app.flags.DEFINE_string('tf_records_train_path', 
                           os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'encoder/data/tf_records/train/')),
                           'Path of the training data.')

tf.app.flags.DEFINE_string('tf_records_test_path', 
                           os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'encoder/data/tf_records/test/')),
                           'Path of the test data.')

tf.app.flags.DEFINE_string('checkpoints_path', os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'checkpoints/model.ckpt')), 
                           'Path for the test data.')

tf.app.flags.DEFINE_integer('num_epoch', 10,
                            'Number of training epochs.')

tf.app.flags.DEFINE_integer('batch_size', 100,
                            'Size of the training batch.')

tf.app.flags.DEFINE_float('learning_rate',0.00001,
                          'Learning_Rate')

tf.app.flags.DEFINE_boolean('l2_reg', True,
                            'L2 regularization.'
                            )
tf.app.flags.DEFINE_float('lambda_',0.01,
                          'Wight decay factor.')

tf.app.flags.DEFINE_integer('num_v', 1000,
                            'Number of visible neurons (Number of movies the users rated.)')

tf.app.flags.DEFINE_integer('num_h', 20,
                            'Number of hidden neurons.)')

tf.app.flags.DEFINE_integer('num_samples', 10000,
                            'Number of training samples (Number of users, who gave a rating).')
tf.app.flags.DEFINE_integer('num_layer1', 256,
                            'Number of layer 1 .')
tf.app.flags.DEFINE_integer('num_layer2', 128,
                            'Number of layer 2 .')

FLAGS = tf.app.flags.FLAGS

def _compute_error(v0,v1):
     mask=tf.where(tf.equal(v0,0.0), tf.zeros_like(v0), v0)
     num_label=tf.cast(tf.count_nonzero(mask),dtype=tf.float32)
     bool_mask=tf.cast(mask,dtype=tf.bool) 
     outputs=tf.where(bool_mask, v1, tf.zeros_like(v1))
     MSE_loss=tf.cond(num_label>0,lambda:tf.div(tf.reduce_sum(tf.square(tf.subtract(v0,outputs))),num_label),lambda:0.0)
     RMSE_loss=tf.sqrt(MSE_loss)
     return RMSE_loss
    
def main(_):
    with tf.Graph().as_default():
    
        hiddenUnits = FLAGS.num_h
        visibleUnits = FLAGS.num_v
        
        train_data, train_data_infer=_get_training_data(FLAGS)
        test_data=_get_test_data(FLAGS)
        
        iter_train = train_data.make_initializable_iterator()
        iter_train_infer=train_data_infer.make_initializable_iterator()
        iter_test=test_data.make_initializable_iterator()
        
        x_train= iter_train.get_next()/5
        x_train_infer=iter_train_infer.get_next()/5
        x_test=iter_test.get_next()/5
        
        vb = tf.placeholder(tf.float32, [visibleUnits])  # Number of unique movies
        hb = tf.placeholder(tf.float32, [hiddenUnits])  # Number of features were going to learn
        W = tf.placeholder(tf.float32, [visibleUnits, hiddenUnits])  # Weight Matrix
        v_test=x_test 
        # Phase 1: Input Processing
        v0 = x_train
        _h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)  # Visible layer activation
        h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))  # Gibb's Sampling

        # Phase 2: Reconstruction
        _v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)  # Hidden layer activation
        v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
        h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)
        
        hh0 = tf.nn.sigmoid(tf.matmul(x_train_infer, W) + hb)
        vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
        
        # Learning rate
        alpha = FLAGS.learning_rate

        # Create the gradients
        w_pos_grad = tf.matmul(tf.transpose(v0), h0)
        w_neg_grad = tf.matmul(tf.transpose(v1), h1)

        # Calculate the Contrastive Divergence to maximize
        CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])

        # Create methods to update the weights and biases
        update_w = W + alpha * CD
        update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
        update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

        # Set the error function, here we use Mean Absolute Error Function
        #err = v0 - v1
        #err_sum = tf.reduce_mean(err*err)
        err_sum=_compute_error(v0,v1)
        err_test=_compute_error(v_test,vv1)

        """ Initialize our Variables with Zeroes using Numpy Library """

        # Current weight
        cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

        # Current visible unit biases
        cur_vb = np.zeros([visibleUnits], np.float32)

        # Current hidden unit biases
        cur_hb = np.zeros([hiddenUnits], np.float32)

        # Previous weight
        prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

        # Previous visible unit biases
        prv_vb = np.zeros([visibleUnits], np.float32)

        # Previous hidden unit biases
        prv_hb = np.zeros([hiddenUnits], np.float32)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
   
        epochs = FLAGS.num_epoch
        batchsize = FLAGS.batch_size
        errors = []
        test_loss=[]
        
        num_batches=int(FLAGS.num_samples/FLAGS.batch_size)

        #saver=tf.train.Saver()
        for epoch in range(FLAGS.num_epoch):
                
            sess.run(iter_train.initializer)
            sess.run(iter_train_infer.initializer)
            sess.run(iter_test.initializer)
            for batch_nr in range(num_batches):
                #print(sess.run(x_train))
                cur_w = sess.run(update_w, feed_dict={W: prv_w, vb: prv_vb, hb: prv_hb})
                cur_vb = sess.run(update_vb, feed_dict={ W: prv_w, vb: prv_vb, hb: prv_hb})
                cur_hb = sess.run(update_hb, feed_dict={ W: prv_w, vb: prv_vb, hb: prv_hb})
                prv_w = cur_w
                prv_vb = cur_vb
                prv_hb = cur_hb  
            errors.append(sess.run(err_sum, feed_dict={ W: cur_w, vb: cur_vb, hb: cur_hb}))
            #print(errors[-1])  
            for i in range(FLAGS.num_samples):
                #feed = sess.run(hh0, feed_dict={v0: x_train_infer, W: prv_w, hb: prv_hb})
                #rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})
                loss_,vv1_=sess.run((err_test,vv1), feed_dict={ W: cur_w, vb: cur_vb, hb: cur_hb})
                test_loss.append(loss_)
                #print(vv1_)
            print('epoch_nr: %i, train_loss: %.3f, test_loss: %.3f'
                      %(epoch,(errors[-1]),np.mean(test_loss)))
                
            #if np.mean(test_loss)<1:
                #saver.save(sess, FLAGS.checkpoints_path)


        sess.run(iter_train.initializer)
        sess.run(iter_train_infer.initializer)
        sess.run(iter_test.initializer) 
        x_train_infer=tf.where(tf.equal(x_train_infer,0.0), x_test, x_train_infer)
           
        recs=[]
        for i in range(FLAGS.num_samples):
            #feed = sess.run(hh0, feed_dict={v0: x_train_infer, W: prv_w, hb: prv_hb})
            rec = sess.run(vv1, feed_dict={ W: prv_w, vb: prv_vb,hb:prv_hb})
            recs.append(rec)
            
                
        recs=np.array(recs)
        recs=5*recs.reshape(10000,1000)
        #preds=denormalizeData(preds,3.8572805008190647,4.016329062225046)
        WriteToCSV(recs,path='rbm.csv')
             
if __name__ == "__main__":
    
    tf.app.run()
