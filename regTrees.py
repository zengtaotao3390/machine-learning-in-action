import numpy as np


def loadDataSet(fileName):
    dataMat = []
    with open(fileName) as fr:
        lines = fr.readlines()
    for line in lines:
        curLine = line.strip().split("\t")
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    # 如果是树回归，那么feature是一个常数，如果是模型树，其模型是一个线性方程
    # 所以这个的mat0是一个尝试也可以理解
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


# 建立叶节点函数
def regLeaf(dataSet):#returns the value used for each leaf
    return np.mean(dataSet[:,-1])


# 误差计算函数
def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

# 最佳方式切分数据集和生成相应的叶子节点
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]
    # 当所有值相等，退出
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        None, leafType(dataSet)
    m,n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n -1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestS = newS
                bestIndex = featIndex
                bestValue = splitVal
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


# tesMat = np.mat(np.eye(4))
# mat0, mat1 = binSplitDataSet(tesMat, 1, 0.5)
# print(mat0)
# print(mat1)

# myData = loadDataSet('./machinelearninginaction/Ch09/ex00.txt')
# myMat = np.mat(myData)
# myTree = createTree(myMat)
# print(myTree)
#
# myData = loadDataSet('./machinelearninginaction/Ch09/ex0.txt')
# myMat = np.mat(myData)
# myTree = createTree(myMat)
# print(myTree)
# myTree = createTree(myMat, ops=(0, 1))
# print(myTree)
#
# np.power()


def isTree(obj):
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    if isTree(tree['right']):
       tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    # 如果没有数据落入此分支区域，说明此树与测试数据完全不吻合，那么将进行塌陷，塌陷这个名字取得好
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    # 不是根节点
    if (isTree(tree['left'])) or (isTree(tree['right'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + np.sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['right'] + tree['left']) / 2.0
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree


mydata2 = loadDataSet('./machinelearninginaction/Ch09/ex2.txt')
myMat2 = np.mat(mydata2)
myTree = createTree(myMat2)
print(myTree)
myDataTest = loadDataSet('./machinelearninginaction/Ch09/ex2test.txt')
myMatTest = np.mat(myDataTest)
pruneTree = prune(myTree, myMatTest)
print(pruneTree)
