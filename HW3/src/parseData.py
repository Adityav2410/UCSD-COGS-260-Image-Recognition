from itertools import cycle
import numpy as np
from sklearn.model_selection import train_test_split


class Cifar:
    
    def __init__(self,datapath,batchSize=10,splitRatio=0.2):
        self.datapath = datapath
        self.batchSize = batchSize
        self.splitRatio = splitRatio
        self.fileList = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4' , 'data_batch_5']

        self.trX,self.teX,self.trY,self.teY = self.generateCifarData()
        self.trainGenerator = self.generateTrainData()
        self.testGenerator = self.generateTestData()


    def unpickle(self,file):
        import cPickle
        with open(file, 'rb') as fo:
            dict = cPickle.load(fo)
        return dict

    def generateCifarData(self):
        
        trainX = np.zeros([0,3,32,32])
        trainY = np.zeros([0])    
        for fileName in self.fileList:
            filePath = self.datapath + fileName
            data = self.unpickle(filePath)
            data1 = data.items()
            trX = data1[0][1]
            trY = np.asarray(data1[1][1])
            trX_reshaped = trX.reshape([-1,3,32,32])
            trainX = np.concatenate((trainX, trX_reshaped),axis = 0)
            trainY = np.concatenate((trainY,trY),axis = 0)
        
        trainX = trainX.transpose([0,2,3,1]).astype("uint8")
        trainY = trainY.astype(int)
        
        x_train, x_test, y_train, y_test = train_test_split( trainX, trainY, test_size=self.splitRatio, random_state = 42)
        
        print("Number of train examples: ",x_train.shape[0])
        print("Number of test examples: ",x_test.shape[0])
        return [x_train,x_test, y_train, y_test]


    def generateTrainData(self):

        currIndex = 0;
        data_x = self.trX.tolist()
        data_y = self.trY.tolist()
        
        batchX = np.zeros((0,32,32,3))
        batchY = np.zeros([0,10])
        
        for x,y in cycle( zip(data_x,data_y)):
            x = np.array(x)

            currIndex = currIndex + 1
            
            batchX = np.concatenate( (batchX, x.reshape([1,32,32,3])),axis=0 )
            batchY = np.concatenate( (batchY,np.zeros((1,10))), axis = 0 )
            batchY[-1,y] = 1

            if currIndex == self.batchSize:
                yield np.copy(batchX),np.copy(batchY)
                batchX = np.zeros((0,32,32,3))
                batchY = np.zeros((0,10))
                currIndex       = 0


    def generateTestData(self):

        currIndex = 0;
        data_x = self.teX.tolist()
        data_y = self.teY.tolist()
        
        batchX = np.zeros((0,32,32,3))
        batchY = np.zeros((0,10))
        
        for x,y in cycle( zip(data_x,data_y)):
            x = np.array(x)
            currIndex = currIndex + 1
            
            batchX = np.concatenate( (batchX, x.reshape([1,32,32,3])),axis=0 )
            batchY = np.concatenate( (batchY,np.zeros((1,10))), axis = 0 )
            batchY[-1,y] = 1

            if currIndex == self.batchSize:
                yield np.copy(batchX),np.copy(batchY)
                batchX = np.zeros((0,32,32,3))
                batchY = np.zeros((0,10))
                currIndex       = 0

    def train_next_batch(self,batchSize):
        self.batchSize = batchSize
        return self.trainGenerator.next()

    def test_next_batch(self,batchSize):
        self.batchSize = batchSize
        return self.testGenerator.next()