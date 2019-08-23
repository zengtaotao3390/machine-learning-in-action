from math import log
import operator


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


# myData, labels = createDataSet()
# print(chooseBestFeatureToSplit(myData))
# print(myData)


def majorityCount(classList):
    classCount = {}
    for vote in classCount:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter[1], reverse=True)
    return sortedClassCount[0][0]


# labels 为属性值
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # if classList.count(classList[0]) == len(classList):
    #     return classList[0]
    uniqueClassList = set(classList)
    # 如果为一个类型
    if len(uniqueClassList) == 1:
        for classX in uniqueClassList:
            return classX
    # 只有一个特征属性，返回最大分类
    if len(dataSet[0]) == 1:
        return majorityCount(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    featureLabel = labels[bestFeature]
    tree = {featureLabel: {}}
    del(labels[bestFeature])
    uniqueVal = set([example[bestFeature] for example in dataSet])
    for key in uniqueVal:
        copyLabels = labels[:]
        subDataSet = splitDataSet(dataSet, bestFeature, key)
        tree[featureLabel][key] = createTree(subDataSet, copyLabels)
    return tree


# dataSet, labels = createDataSet()
# print(createTree(dataSet, labels))


