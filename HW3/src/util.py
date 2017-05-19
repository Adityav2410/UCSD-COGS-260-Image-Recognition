import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


def getData(datapath):
    trainDataPath = datapath + 'iris_train.data'
    testDataPath =  datapath + 'iris_test.data'

    trainData = pd.read_csv(trainDataPath, sep=",", header = None)
    trainData.columns = ["sepal Length", "sepal Width", "petal Length", "petal Width", "className"]
    trainData['classLabel'] = (trainData['className'] == 'Iris-setosa').astype(int)

    testData = pd.read_csv(testDataPath, sep=",", header = None)
    testData.columns = ["sepal Length", "sepal Width", "petal Length", "petal Width", "className"]
    testData['classLabel'] = (testData['className'] == 'Iris-setosa').astype(int)

    print "Number of training example: \t", trainData.shape[0]
    print "Number of test example: \t", testData.shape[0],"\n"

    featureLabels = list(trainData)[0:4]
    print "The features in the dataset are:"
    print featureLabels,"\n"
    
    print 'Class label for Iris-setosa: \t\t 1'
    print 'Class label for Iris-versicolor: \t 0'
    
    return trainData.round(2), testData.round(2), featureLabels


def createTiles(x=1,y=1,hwidth=8,vwidth=4): 
    fig,plots = plt.subplots(x,y,figsize=(hwidth,vwidth));
    plots = plots.flatten()
    return(fig, plots)


def scatterFeatures(featureLabels, i, j, trainData, plots, plotIndex):

    feature1_name = featureLabels[i]
    feature2_name = featureLabels[j]

    class1_index = trainData[trainData['classLabel'] == 0].index
    class2_index = trainData[trainData['classLabel'] == 1].index

    class1_feature1_coord = trainData.iloc[class1_index][feature1_name]
    class1_feature2_coord = trainData.iloc[class1_index][feature2_name]

    class2_feature1_coord = trainData.iloc[class2_index][feature1_name]
    class2_feature2_coord = trainData.iloc[class2_index][feature2_name]

    label1 = plots[plotIndex].scatter(class1_feature1_coord,class1_feature2_coord, marker='x', color='r',label='Iris-versicolor')
    label2 = plots[plotIndex].scatter(class2_feature1_coord,class2_feature2_coord, marker='o', color='b',label='Iris-setosa')

    plots[plotIndex].set_title(feature1_name + '  vs  ' + feature2_name )
    plots[plotIndex].set_xlabel(feature1_name)
    plots[plotIndex].set_ylabel(feature2_name)

    plots[plotIndex].legend(handles=[label1, label2])    