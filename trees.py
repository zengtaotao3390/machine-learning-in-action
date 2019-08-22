from math import log

def calcShannonEnt(dataSet):
    numEntities = len(dataSet)
    labelCount = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1
    shannonEnt = 0.0
    # 出现该分类的概率
    for label in labelCount.keys():
        px = float(labelCount[label]) / numEntities
        lx = - log(px, 2)
        shannonEnt += px * lx
    return shannonEnt


def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'floppers']
    return dataSet, labels


# mydata, labels = createDataSet()
# print(calcShannonEnt(mydata))
# mydata[0][-1] = 'maybe'
# print(calcShannonEnt(mydata))


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featureVector in dataSet:
        if featureVector[axis] == value:
            reducedFeatureVector = featureVector[: axis]
            reducedFeatureVector.extend(featureVector[axis + 1:])
            retDataSet.append(reducedFeatureVector)
    return retDataSet


def calcBestFeature(dataSet):
    baseEntropy = calcShannonEnt(dataSet)
    featureNum = len(dataSet[0]) - 1
    for i in range(featureNum):
        keys = set(dataSet[:][i])
        for key in keys:
            subDataSet = createDataSet()