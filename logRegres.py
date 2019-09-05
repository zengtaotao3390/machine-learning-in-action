from numpy import *
import matplotlib.pyplot as plt

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


# dataSet, classLabel = loadData()
# print(gradientAscent(dataSet, classLabel))

def plotBestFit():
    aPointX = []; aPointY = []
    bPointX= []; bPointY = []
    dataSet, classLabel = loadData()
    rowNum = shape(dataSet)[0]
    weights = gradientAscent(dataSet, classLabel).A
    for i in range(rowNum):
        if int(classLabel[i]) == 1:
            aPointX.append(dataSet[i][1])
            aPointY.append(dataSet[i][2])
        else:
            bPointX.append(dataSet[i][1])
            bPointY.append(dataSet[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(aPointX, aPointY, s=30, c='red', marker='s')
    ax.scatter(bPointX, bPointY, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    plt.plot(x, y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

plotBestFit()