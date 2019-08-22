from math import log

def calcShannonEnt(dataSet):
    numEntities = len(dataSet)
    labelCount = {}
    for entity in dataSet:
        entityLabel = entity[-1]
        if entityLabel not in labelCount.keys():
            labelCount[entityLabel] = 0
        labelCount[entityLabel] += 1
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


mydata, labels = createDataSet()
print(calcShannonEnt(mydata))