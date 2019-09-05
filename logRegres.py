import math
from numpy import *

def loadData():
    with open('./machinelearninginaction/Ch05/testSet.txt') as orgFile:
        lines = orgFile.readlines()
    dataSet = []
    classLable= []
    for line in lines:
        lineSplit = line.split()
        dataSet.append([1.0, float(lineSplit[0]), float(lineSplit[1])])
        classLable.append(int(lineSplit[2]))
    return dataSet, classLable


def sigmoid(intX):
    # use math.exp error
    return 1.0 / (1 + exp(-intX))


def gradientAscent(dataSet, classLabel):
    dataSetMatrix = mat(dataSet)
    classLabelMatrix = mat(classLabel).transpose()
    rowNum, columnNum = dataSetMatrix.shape
    loopNum = 500
    alpha = 0.001
    weights = ones((columnNum, 1))
    for i in range(loopNum):
        preLabel = sigmoid(dataSetMatrix * weights)
        error = classLabelMatrix - preLabel
        weights = weights + alpha * dataSetMatrix.transpose() * error
    return weights


dataSet, classLabel = loadData()
print(gradientAscent(dataSet, classLabel))