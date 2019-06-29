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

tf.app.flags.DEFINE_integer('num_epoch', 20,
                            'Number of training epochs.')

tf.app.flags.DEFINE_integer('batch_size', 16,
                            'Size of the training batch.')

tf.app.flags.DEFINE_float('learning_rate',0.00005,
                          'Learning_Rate')
tf.app.flags.DEFINE_float('learning_rate_decay',0.9,
                          'Learning_Rate decay')

tf.app.flags.DEFINE_boolean('l2_reg', True,
                            'L2 regularization.'
                            )
tf.app.flags.DEFINE_float('lambda_',0.001,
                          'Wight decay factor.')
tf.app.flags.DEFINE_float('drop_out_prob',0.5,
                          'drop out prob.')

tf.app.flags.DEFINE_integer('num_v', 1000,
                            'Number of visible neurons (Number of movies the users rated.)')

tf.app.flags.DEFINE_integer('num_h', 1000,
                            'Number of hidden neurons.)')

tf.app.flags.DEFINE_integer('num_samples', 10000,
                            'Number of training samples (Number of users, who gave a rating).')
tf.app.flags.DEFINE_integer('num_layer1', 256,
                            'Number of layer 1 .')
tf.app.flags.DEFINE_integer('num_layer2', 128,
                            'Number of layer 2 .')

tf.app.flags.DEFINE_boolean('constrain', True,
                            'if constrained.')
tf.app.flags.DEFINE_boolean('batch_normalization', True,
                            'batch_normalization.')
FLAGS = tf.app.flags.FLAGS
def main(_):
    '''Building the graph, opening of a session and starting the training od the neural network.'''
    
    num_batches=int(FLAGS.num_samples/FLAGS.batch_size)

    with tf.Graph().as_default():

        train_data, train_data_infer=_get_training_data(FLAGS)
        test_data=_get_test_data(FLAGS)
        
        iter_train = train_data.make_initializable_iterator()
        iter_train_infer=train_data_infer.make_initializable_iterator()
        iter_test=test_data.make_initializable_iterator()
        
        x_train= iter_train.get_next()
        x_train_infer=iter_train_infer.get_next()
        x_test=iter_test.get_next()

        model=TrainModel(FLAGS, 'training')

        train_op, train_loss_op,outputs_op=model.train(x_train)
        prediction, labels, test_loss_op, mae_ops=model._validation_loss(x_train_infer, x_test)
        prediction_test=model._test_loss(x_train_infer, x_test)

        saver=tf.train.Saver()
        
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())
            train_loss=0
            test_loss=[]
            mae=[]
            preds=[]

            for epoch in range(FLAGS.num_epoch):
                
                sess.run(iter_train.initializer)
                sess.run(iter_train_infer.initializer)
                sess.run(iter_test.initializer)
                
                for batch_nr in range(num_batches):
                    
                    x_train_,_, loss_,outputs_=sess.run((x_train,train_op, train_loss_op,outputs_op))
                    
                    train_loss+=loss_
                    #print(x_train_)
                    #print(outputs_)
                    #print(loss_)
                    
                
                for i in range(FLAGS.num_samples):
                    
                    pred, labels_, loss_, mae_=sess.run((prediction, labels, test_loss_op,mae_ops))
                    #print(loss_)
                    #print(mae_)
                    #print(pred)
                    #print(labels)
                    if loss_>0:
                        test_loss.append(loss_)
                        mae.append(mae_)
                
                sess.run(model.learning_rate_op)


                    
                    
                print('epoch_nr: %i, train_loss: %.3f, test_loss: %.3f, mean_abs_error: %.3f'
                      %(epoch,(train_loss/num_batches),np.sqrt(np.mean(test_loss)), np.mean(mae)))
                
                if np.mean(mae)<1:
                    saver.save(sess, FLAGS.checkpoints_path)

                train_loss=0
                test_loss=[]
                mae=[]
            sess.run(iter_train.initializer)
            sess.run(iter_train_infer.initializer)
            sess.run(iter_test.initializer) 
            #x_train_infer=tf.where(tf.equal(x_train_infer,0.0), x_test, x_train_infer)
            
            for i in range(FLAGS.num_samples):
                
                pred=sess.run(prediction_test)
                preds.append(pred)
                
            preds=np.array(preds)
            preds=preds.reshape(10000,1000)
            #preds=denormalizeData(preds,3.8572805008190647,4.016329062225046)
            #WriteToCSV(preds,path='encoder.csv')
            WriteToCSV(preds, path='cache/default', sample=parameters.VALTRUTH_PATH)
            WriteToCSV(preds, path='test/default', sample=parameters.SAMPLECSV_PATH)
                    
if __name__ == "__main__":
    
    tf.app.run()
