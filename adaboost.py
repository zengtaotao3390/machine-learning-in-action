from numpy import *


def loadSimpData():
    dataMat = matrix([[1., 2.1],
                      [2., 1.1],
                      [1.3, 1.],
                      [1., 1.],
                      [2., 1.]
                      ])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


def stumpClassify(dataMatrix, dimension, thresholdValue, thresholdIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if thresholdIneq == 'lt':
        retArray[dataMatrix[:, dimension] <= thresholdValue] = -1.0
    else:
        retArray[dataMatrix[:, dimension] > thresholdValue] = -1.0
    return retArray


def bulidStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMatrix = mat(classLabels).T
    m, n = shape(dataArr)
    numStemps = 10.0
    bestStump = {}
    bestClassEstimate = mat(zeros((m, 1)))
    # 无穷大
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numStemps
        for j in range(-1, int(numStemps) + 1):
            for inequal in ['lt', 'gt']:
                thresholdValue = rangeMin + stepSize * j
                predictedVals = stumpClassify(dataMatrix, i, thresholdValue, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMatrix] = 0
                weightedError = D.T * errArr
                print('split: dim {}, threshold {}, threshold ineqal: {}, the weighted error is {}'.format(i, thresholdValue, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEstimate = predictedVals.copy()
                    bestStump['dimension'] = i
                    bestStump['threshold'] = thresholdValue
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEstimate


# D = mat(ones((5, 1)) / 5)
# dataMat, classLabels = loadSimpData()
# print(bulidStump(dataMat, classLabels, D))


def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)
    aggregateClassEstimate = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEstimate = bulidStump(dataArr, classLabels, D)
        print('D: ', D.T)
        # 防止下溢除0
        alpha = float(0.5 * log((1 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('classEstimate: ', classEstimate.T)
        # 预测正确的为1， 错误的为-1
        expon = multiply(-1 * alpha * mat(classLabels).T, classEstimate)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggregateClassEstimate += alpha * classEstimate
        print("aggregateClassEstimate: ", aggregateClassEstimate.T)
        aggregateErrors = multiply(sign(aggregateClassEstimate) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggregateErrors.sum() / m
        print('total error: ', errorRate, "\n")
        if errorRate == 0.0:
            break
    return weakClassArr, aggregateClassEstimate

# dataMat, classLabels = loadSimpData()
# classifierArray = adaBoostTrainDS(dataMat, classLabels, 9)
# print(classifierArray)


def adaClassify(dataToClass, classifierArray):
    dataMat = mat(dataToClass)
    m = shape(dataMat)[0]
    aggregateEstimate = mat(zeros((m ,1)))
    for i in range(len(classifierArray)):
        classEstimate = stumpClassify(dataMat, classifierArray[i]['dimension'], classifierArray[i]['threshold'], classifierArray[i]['ineq'])
        aggregateEstimate += classifierArray[i]['alpha'] * classEstimate
        print(aggregateEstimate)
    return sign(aggregateEstimate)


# dataMat, classLabels = loadSimpData()
# classifierArray = adaBoostTrainDS(dataMat, classLabels, 9)
# predictLabel = adaClassify([0, 0], classifierArray)
# print(predictLabel)
# predictLabel = adaClassify([[5, 5], [0, 0]], classifierArray)
# print(predictLabel)


def plotRoc(classLabels, aggregateEstimate):
    import matplotlib.pyplot as plt
    posSampleNum = sum(array(classLabels) == 1.0)
    cursor = (1.0, 1.0)
    yStep = float(1 / posSampleNum)
    xStep = float(1 / (len(classLabels) - posSampleNum))
    ySum = 0.0
    sortedAggregateEstimate = aggregateEstimate.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedAggregateEstimate.tolist()[0]:
        if classLabels[index] == 1:
            xDel = 0
            yDel = yStep
        else:
            xDel = xStep
            yDel = 0
            ySum += cursor[1] * xStep
        ax.plot([cursor[0], cursor[0] - xDel], [cursor[1], cursor[1] - yDel], c='b')
        cursor = (cursor[0] - xDel, cursor[1] - yDel)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print('the Area Under the Curve is :{}', ySum)


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat




dataMat, classLabels = loadDataSet('./machinelearninginaction/Ch07/horseColicTraining2.txt')
classifierArray, aggregateClassEstimate = adaBoostTrainDS(dataMat, classLabels, 9)
plotRoc(classLabels, aggregateClassEstimate.T)
