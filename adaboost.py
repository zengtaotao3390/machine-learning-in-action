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
    return weakClassArr

# dataMat, classLabels = loadSimpData()
# classifierArray = adaBoostTrainDS(dataMat, classLabels, 9)
# print(classifierArray)


