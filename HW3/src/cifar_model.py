import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split



# Define Variables - Bias and Weights
def weight_variable(shape,name="weight"):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return( tf.Variable(initial,name=name) )

def bias_variable(shape,name="bias"):
    initial = tf.constant(0.1,shape = shape)
    return( tf.Variable(initial,name=name) )

# Define Convolution, maxPooling and BatchNorm functions
def conv2d(x,W,name="conv2d"):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1,],padding='SAME',name=name)

def max_pool_2x2(x,name="maxpool"):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],
                         strides=[1,2,2,1],padding='SAME',name=name)
def batch_norm(x, is_training=True,name="batch_norm"):
    return slim.batch_norm(x, decay= 0.9, updates_collections= None, is_training= is_training,name=name)


def getTrainStep(cross_entropy,optimizerDict):
	typeOptimizer = optimizerDict['type']
	lr = optimizerDict['lr']
	momentum = optimizerDict['momentum']

	if typeOptimizer == 'SGD':
		train_step = tf.train.GradientDescentOptimizer(learning_rate = lr).minimize(cross_entropy)
	if typeOptimizer == 'Nesterov':
		train_step = tf.train.MomentumOptimizer(learning_rate = lr, momentum = momentum, use_nesterov=True).minimize(cross_entropy)
	if typeOptimizer == 'RMSprop':
		train_step = tf.train.RMSPropOptimizer(learning_rate = lr, decay=0.9, momentum=momentum).minimize(cross_entropy)
	if typeOptimizer == 'AdaGrad':
		train_step = tf.train.AdagradOptimizer(learning_rate=lr)
	return(train_step)


def buildFCmodel(optimizerDict,performBN = False):
	startTime = time.time()
	x = tf.placeholder(tf.float32,shape=[None,32,32,3],name="input_image")
	y_true = tf.placeholder(tf.float32,shape=[None,10], name="true_label")
	keep_prob = tf.placeholder(tf.float32)

	# Layer 1
	W_conv1 = weight_variable([3,3,3,32],name="weight_conv_1")
	b_conv1 = bias_variable([32],name="bias_conv_1")
	h_conv1 = tf.nn.relu(conv2d(x,W_conv1)+b_conv1,name="relu_conv_1")
	h_pool1 = max_pool_2x2(h_conv1,name="maxpool_conv_1")
	h_pool1_bn = batch_norm(h_pool1) if performBN else h_pool1

	# Layer 2
	W_conv2 = weight_variable([3,3,32,64],name="weight_conv_2")
	b_conv2 = bias_variable([64],name="bias_conv_2")
	h_conv2 = tf.nn.relu(conv2d(h_pool1_bn,W_conv2)+b_conv2,name="relu_conv_2")
	h_pool2 = max_pool_2x2(h_conv2,name="maxpool_conv2")
	h_pool2_bn = batch_norm(h_pool2) if performBN else h_pool2


	# Layer 3
	W_conv3 = weight_variable([3,3,64,64],name="weight_conv3")
	b_conv3 = bias_variable([64],name="bias_conv3")
	h_conv3 = tf.nn.relu(conv2d(h_pool2_bn,W_conv3)+b_conv3,name="relu_conv3")
	h_pool3 = max_pool_2x2(h_conv3,name="maxpool_conv3")
	h_pool3_bn = batch_norm(h_pool3) if performBN else h_pool3

	# Flatten last conv feature layer
	h_pool3_flat = tf.reshape(h_pool3_bn,[-1,4*4*64],name="flatten")
	h_pool3_flat_bn = batch_norm(h_pool3_flat) if performBN else h_pool3_flat

	# Fully Connected layer1
	W_fc1 = weight_variable([4*4*64,256],name="fc1_weight")
	b_fc1 = bias_variable([256],name="fc1_bias")
	h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat_bn,W_fc1) + b_fc1,name="fc1_relu")
	h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob,name="fc1_dropout")
	h_fc1_bn = batch_norm(h_fc1_drop) if performBN else h_fc1_drop

	# Fully connected layer 2
	W_fc2 = weight_variable([256,64],name="fc2_weight")
	b_fc2 = bias_variable([64],name="fc2_bias")
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1_bn,W_fc2) + b_fc2,name="fc2_relu")
	h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob,name="fc2_dropout")

	# Readout Layer
	W_fc3 = weight_variable([64,10],name="out_layer_weight")
	b_fc3 = bias_variable([10],name="outLayer_bias")
	y_pred = tf.matmul(h_fc2_drop,W_fc3) + b_fc3

	# Define the loss and other optimization steps
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits=y_pred))
	train_step = getTrainStep(cross_entropy,optimizerDict)
	correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

	endTime = time.time()
	getTrainableParameters()
	print("Time to build the network {:.2f}s".format(endTime-startTime) )

	fcModelParam = {}
	fcModelParam['x'] = x
	fcModelParam['y_true'] = y_true
	fcModelParam['y_pred'] = y_pred
	fcModelParam['keep_prob'] = keep_prob
	fcModelParam['loss'] = cross_entropy
	fcModelParam['train_step'] = train_step
	fcModelParam['accuracy'] = accuracy
	return(fcModelParam)
	


#sess.run(tf.global_variables_initializer())

def trainModel(sess,modelParam,cifar,keep_prob = 1,iter=500, batchSize = 128):
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy  = []

    print("Training Network")
    for i in range(iter):
        trX, trY = cifar.train_next_batch(batchSize)

        if( i%10 == 0):
            epoc_loss,epoc_accuracy = sess.run(  [modelParam['loss'],modelParam['accuracy']],      \
                                                    feed_dict={modelParam['x']:trX,modelParam['y_true']:trY,modelParam['keep_prob']:keep_prob}
                                                 )
            train_loss.append(epoc_loss)
            train_accuracy.append(epoc_accuracy)
            
            teX,teY = cifar.test_next_batch(batchSize)
            epoc_loss,epoc_accuracy = sess.run(  [modelParam['loss'],modelParam['accuracy']],      \
                                                    feed_dict={modelParam['x']:teX,modelParam['y_true']:teY,modelParam['keep_prob']:1}
                                                 )
            test_loss.append(epoc_loss)
            test_accuracy.append(epoc_accuracy)
            
            print("Step %d, \t Accuracy: %.2f , \t Train Loss: %.2f, \t Test Loss: %.2f "%(i,train_accuracy[-1],train_loss[-1],test_loss[-1]))

        modelParam['train_step'].run(feed_dict={modelParam['x']:trX,modelParam['y_true']:trY,modelParam['keep_prob']:keep_prob})

    print("Test Complete.")
    teX,teY = cifar.test_next_batch(1000)
    print(" Accuracy: %g"%modelParam['accuracy'].eval(feed_dict={modelParam['x']:teX,modelParam['y_true']:teY,modelParam['keep_prob']:1.0}))

    legend1, = plt.plot(train_loss,label='Train loss')
    legend2, = plt.plot(test_loss,label='Test loss')
    plt.xlabel('Iterations(x100)')
    plt.ylabel('Loss')
    plt.title('CNN training: loss vs epocs')
    plt.legend()
    plt.show()


def predict(sess,modelParam,x,y_true):
	[y_pred,accuracy] = sess.run([modelParam['y_pred'],modelParam['accuracy']],feed_dict={modelParam['x']:x,modelParam['y_true']:y_true,modelParam['keep_prob']:1})
	print("Accuracy:", accuracy)
	y_pred_class = np.argmax(y_pred,1)
	print("Predicted class: ",y_pred_class)
	print("True class: ",y_true)
	return(y_pred_class)



def getTrainableParameters():
	total_parameters = 0
	for variable in tf.trainable_variables():
	    # shape is an array of tf.Dimension
	    shape = variable.get_shape()
	    #print(shape)
	    #print(len(shape))
	    variable_parametes = 1
	    for dim in shape:
	        #print(dim)
	        variable_parametes *= dim.value
	    #print(variable_parametes)
	    total_parameters += variable_parametes
	print("Total number of trainable parameters: ",total_parameters)