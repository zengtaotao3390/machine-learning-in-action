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


def chooseBestFeatureToSplit(dataSet):
    baseEntropy = calcShannonEnt(dataSet)
    featureNum = len(dataSet[0]) - 1
    bestInfoGain = 0.0
    bestFeatureAxis = -1
    for i in range(featureNum):
        entropyCount = 0.0
        featureList = [example[i] for example in dataSet]
        uniqueVals = set(featureList)
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            pro = len(subDataSet) / float(len(dataSet))
            subDataSetShannonEntropy = calcShannonEnt(subDataSet)
            entropyCount += pro * subDataSetShannonEntropy
        infoGain = baseEntropy - entropyCount
        if(bestInfoGain < infoGain):
            bestInfoGain = infoGain
            bestFeatureAxis = i
    return bestFeatureAxis


myData, labels = createDataSet()
print(chooseBestFeatureToSplit(myData))
print(myData)
