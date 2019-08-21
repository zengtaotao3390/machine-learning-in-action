from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # 数据从小到大的索引值
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(0), reverse=True)
    return sortedClassCount[0][0]


# group, labels = createDataSet()
# print(classify0([0, 0], group, labels, 3))

def file2matrix(filename):
    with open(filename) as fr:
        arrayOflines = fr.readlines()
    numberOflines = len(arrayOflines)
    returnMat = zeros((numberOflines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOflines:
        line.strip()
        listFormLine = line.split('\t')
        returnMat[index, :] = listFormLine[0:3]
        classLabelVector.append(int(listFormLine[-1]))
        index += 1
    return returnMat, classLabelVector


def showLine2Line3():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    datingDataMat, datingDataLabels = file2matrix('./machinelearninginaction/Ch02/datingTestSet2.txt')
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*array(datingDataLabels), 15.0*array(datingDataLabels))
    plt.show()


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    lineNum = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (lineNum, 1))
    normDataSet = normDataSet / tile(ranges, (lineNum, 1))
    return normDataSet, ranges, minVals


# datingDataMat, datingDataLabels = file2matrix('./machinelearninginaction/Ch02/datingTestSet2.txt')
# normDataSet, ranges, minVals = autoNorm(datingDataMat)
# print(datingDataMat)
# print(normDataSet)
# print(ranges)
# print(minVals)


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('./machinelearninginaction/Ch02/datingTestSet2.txt')
    normDataMat, ranges, minVals = autoNorm(datingDataMat)
    m = normDataMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normDataMat[i, :], normDataMat[numTestVecs: m, :], datingLabels[numTestVecs: m], 3)
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print('the total error rate is : %f' % (errorCount / float(numTestVecs)))

# datingClassTest()

def classifyPerson():
    resultList = ['not at all', 'is small doses', 'in large doses']
    ffMilles = float(input('frequent flier miles earned per year?'))
    percentTats = float(input('percentage of time spent playing video games?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataSet, datingLabels = file2matrix('./machinelearninginaction/Ch02/datingTestSet2.txt')
    normDataSet, ranges, minVals = autoNorm(datingDataSet)
    inArr = array([ffMilles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normDataSet, datingLabels, 3)
    print(classifierResult)
    print('you will probable like this person: ', resultList[classifierResult -1])


classifyPerson()