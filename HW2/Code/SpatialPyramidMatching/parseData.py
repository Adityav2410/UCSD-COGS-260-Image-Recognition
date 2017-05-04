import numpy as np
from mnist import MNIST
from PIL import Image

def getMnistData(dataPath, trainSize=50000,validSize=10000,testSize=20000):
        
    # Load data and parse into usable numpy format
    mndata = MNIST(dataPath);
    
    train_Data = mndata.load_training()
    test_Data = mndata.load_testing()
    
    # Convert data to numpy format
    trainX = np.array(train_Data[0])
    trainY = np.array(train_Data[1])
    testX = np.array(test_Data[0])
    testY = np.array(test_Data[1])
    
    # Create data of given sizes
    validX = trainX[0:validSize];
    validY = trainY[0:validSize];
    trainX = trainX[validSize:validSize+trainSize]
    trainY = trainY[validSize:validSize+trainSize]
    testX = testX[0:testSize]
    testY = testY[0:testSize]
    
    # Normalize the data
#     normalize_mean = np.mean(trainX,0)
#     normalize_std  = np.std(trainX-normalize_mean,0)+0.0001
    
#     trainX = np.true_divide(trainX - normalize_mean , normalize_std)
#     validX = np.true_divide(validX - normalize_mean , normalize_std)
#     testX =  np.true_divide(testX  - normalize_mean , normalize_std)
    
    return [trainX,trainY,validX,validY,testX,testY]

# Get data in the format required for SPM code
def getMnistDataForSPM(dataPath):
    trainX,trainY,validX,validY,testX,testY = getMnistData(dataPath, 10000,0,20000)

    trainX_List = []
    trainY_List = []
    for i,image in enumerate(trainX):
        image = trainX[i]
        trainX_List.append(image.reshape([28,28]).astype('uint8'))
        trainY_List.append(str(trainY[i]))

    testX_List = []
    testY_List = []
    for i,image in enumerate(testX):
        image = testX[i]
        testX_List.append(image.reshape([28,28]).astype('uint8'))
        testY_List.append(str(testY[i]))

    print("No of training Examples: ", len(trainX_List))
    print("No of test Examples: ", len(testX_List))
    return trainX_List, trainY_List, testX_List, testY_List
