
# coding: utf-8

# ## Import Libraries

# In[1]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from parseData import Cifar
from cifar_model import buildFCmodel,trainModel


# ## Prepare dataset handle

# In[2]:

dataPath = '/home/adityav/UCSD/Spring17/COGS260_ImageRecognition/HW3/data/cifarData/'
cifar = Cifar(dataPath,batchSize = 10, splitRatio = 0.2)


# ## Config

# In[3]:

performBN                  =   False # performBN(batch normalization) : True to perform Batch normalization else False

optimizerDict = {}
optimizerDict['type']      =  'SGD'  #  'SGD','Nesterov','RMSprop', AdaGrad
optimizerDict['lr']        =   0.5
optimizerDict['momentum']  =   0   # no relevance for SGD/AdaGrad`

dropOut                    =   0.5


# ## Build FC Model

# In[ ]:

fcModelParam = buildFCmodel(optimizerDict,performBN = False)
#fcModelParam->dictionary
#keys: x,y_true,y_pred,keep_prob,loss,train_step,accuracy


# In[ ]:

nIter = 500
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    trainModel(sess,fcModelParam,cifar,keep_prob = dropOut, iter = nIter, batchSize = 10)

