from base_model import BaseModel
import tensorflow as tf


class TrainModel(BaseModel):
    
    def __init__(self, FLAGS, name_scope):
        
        super(TrainModel,self).__init__(FLAGS)
        self.learning_rate = tf.Variable( float(self.FLAGS.learning_rate), trainable=False, dtype=tf.float32 )   # Learning rate.
        self.learning_rate_op=self.learning_rate.assign( self.learning_rate * self.FLAGS.learning_rate_decay )
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self._init_parameters()
        
        
    def _compute_loss(self, predictions, labels,num_labels):
        ''' Computing the Mean Squared Error loss between the input and output of the network.
    		
    	  @param predictions: predictions of the stacked autoencoder
    	  @param labels: input values of the stacked autoencoder which serve as labels at the same time
    	  @param num_labels: number of labels !=0 in the data set to compute the mean
    		
    	  @return mean squared error loss tf-operation
    	  '''
            
        with tf.name_scope('loss'):
            loss_op= tf.cond(num_labels>0,lambda:tf.div(tf.reduce_sum(tf.square(tf.subtract(predictions,labels))),num_labels),lambda:0.0)
            
            return loss_op
    
    def _validation_loss(self, x_train, x_test):
        
        ''' Computing the loss during the validation time.
    		
    	  @param x_train: training data samples
    	  @param x_test: test data samples
    		
    	  @return networks predictions
    	  @return root mean squared error loss between the predicted and actual ratings
    	  '''
        
        outputs=self.inference(x_train,is_training=False) # use training sample to make prediction
        mask=tf.where(tf.equal(x_test,0.0), tf.zeros_like(x_test), x_test) # identify the zero values in the test ste
        num_test_labels=tf.cast(tf.count_nonzero(mask),dtype=tf.float32) # count the number of non zero values
        bool_mask=tf.cast(mask,dtype=tf.bool) 
        outputs=tf.where(bool_mask, outputs, tf.zeros_like(outputs))
    
        MSE_loss=self._compute_loss(outputs, x_test, num_test_labels)
        #RMSE_loss=tf.sqrt(MSE_loss)
        ab_ops= tf.cond(num_test_labels>0,lambda:tf.div(tf.reduce_sum(tf.abs(tf.subtract(x_test,outputs))),num_test_labels),lambda:0.0)
        
            
        return outputs, x_test, MSE_loss, ab_ops
    
    def _test_loss(self, x_train, x_test):
        
        ''' Computing the predicting result.
    		
    	  @param x_train: training data samples
    	  @param x_test: test data samples
    		
    	  @return networks predictions
    	  '''
        
        outputs=self.inference(x_train,is_training=False) # use training sample to make prediction
       
            
        return outputs
    
    def train(self, x):
        '''Optimization of the network parameter through stochastic gradient descent.
            
            @param x: input values for the stacked autoencoder.
            
            @return: tensorflow training operation
            @return: ROOT!! mean squared error
        '''
        
        outputs=self.inference(x,is_training=True)
        mask=tf.where(tf.equal(x,0.0), tf.zeros_like(x), x) # indices of 0 values in the training set
        num_train_labels=tf.cast(tf.count_nonzero(mask),dtype=tf.float32) # number of non zero values in the training set
        bool_mask=tf.cast(mask,dtype=tf.bool) # boolean mask
        outputs=tf.where(bool_mask, outputs, tf.zeros_like(outputs)) # set the output values to zero if corresponding input values are zero

        MSE_loss=self._compute_loss(outputs,x,num_train_labels)
        
        if self.FLAGS.l2_reg==True:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            MSE_loss = MSE_loss +  self.FLAGS.lambda_ * l2_loss
        
        
        #train_op=tf.train.AdagradOptimizer(self.learning_rate).minimize(MSE_loss)
        #train_op=tf.train.MomentumOptimizer(self.learning_rate,momentum=0.9).minimize(MSE_loss)
        #train_op=tf.train.AdagradOptimizer(self.learning_rate).minimize(MSE_loss)
        #train_op=tf.train.AdagradOptimizer(self.learning_rate).minimize(MSE_loss)
        #train_op=tf.train.AdadeltaOptimizer(self.learning_rate).minimize(MSE_loss)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            params = tf.trainable_variables()
            gradients = optimizer.compute_gradients(MSE_loss)
            grads, variables = zip(*gradients)
            grads, _ = tf.clip_by_global_norm(grads, 5)
            parameter_update = optimizer.apply_gradients(zip(grads, variables), global_step=self.global_step)
        #train_op=tf.train.AdamOptimizer(self.learning_rate).minimize(MSE_loss)

        RMSE_loss=tf.sqrt(MSE_loss)

        return parameter_update, RMSE_loss,outputs