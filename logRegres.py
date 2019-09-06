from numpy import *
import matplotlib.pyplot as plt
import random

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


def stochasticGradientAscent(dataSet, classLabel):
    rowNum, columnNum = shape(dataSet)
    alpha = 0.01
    weights = ones(columnNum)
    for i in range(rowNum):
        preLabel = sigmoid(sum(dataSet[i] * weights))
        error = classLabel[i] - preLabel
        print(dataSet[i])
        weights = weights + alpha * error * dataSet[i]
    return weights

def stochasticGradientAscent1(dataSet, clasLabel, iterNum=550):
    rowNum, columnNum = shape(dataSet)
    weights = ones(columnNum)
    for i in range(iterNum):
        indexList = list(range(rowNum))
        for j in range(rowNum):
            alpha = 4 / (1.0 + i + j) + 0.01
            index = int(random.uniform(0, len(indexList)))
            preLabel = sigmoid(sum(dataSet[index] * weights))
            error = clasLabel[index] - preLabel
            weights = weights+ alpha * error * dataSet[index]
            del(indexList[index])
    return weights

# dataSet, classLabel = loadData()
# print(gradientAscent(dataSet, classLabel))

def plotBestFit():
    aPointX = []; aPointY = []
    bPointX= []; bPointY = []
    dataSet, classLabel = loadData()
    rowNum = shape(dataSet)[0]
    # python 的数组要转换成numpy的的数组才能操作
    weights = stochasticGradientAscent1(array(dataSet), classLabel)
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

# plotBestFit()

def logisticClassify(inX, weights):
    prob =  sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def trainLogisticWeights():
    trainLines = []; testLines = []
    trainDataSet = [];trainLabels = []
    with open('./machinelearninginaction/Ch05/horseColicTraining.txt') as trainFile:
        trainLines = trainFile.readlines()
    with open('./machinelearninginaction/Ch05/horseColicTest.txt') as testFile:
        testLines = testFile.readlines()
    for trainLine in trainLines:
        lineSplitDatas = trainLine.strip().split('\t')
        trainData = []
        for i in range(21):
            trainData.append(float(lineSplitDatas[i]))
        trainDataSet.append(trainData)
        trainLabels.append(float(lineSplitDatas[-1]))
    weithts = stochasticGradientAscent1(array(trainDataSet), trainLabels, 1000)
    errorCount = 0
    #   计算错误率
    for testLine in testLines:
        testSplitDatas = testLine.strip().split('\t')
        testData = []
        for i in range(21):
            testData.append(float(testSplitDatas[i]))
        predictionLabel = logisticClassify(testData, weithts)
        if int(predictionLabel) != int(testSplitDatas[-1]):
            errorCount += 1
    errorRate = float(errorCount) / len(testLines)
    print('iteration error rate: {}'.format(errorRate))
    return errorRate

def testAverageRate():
    iterationNum = 10
    errorRateCount = 0.0
    for i in range(iterationNum):
        errorRate = trainLogisticWeights()
        errorRateCount += errorRate
    averageRate = errorRateCount / (iterationNum)
    print('average errrro rage: {}'.format(averageRate))

testAverageRate()
