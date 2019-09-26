import numpy as np


def loadDataSet(fileName):
    dataMat = []
    with open(fileName) as fr:
        lines = fr.readlines()
    for line in lines:
        curLine = line.strip().split("\t")
        fltLine = map(float, curLine)
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
    pass


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




tesMat = np.mat(np.eye(4))`
mat0, mat1 = binSplitDataSet(tesMat, 1, 0.5)
print(mat0)
print(mat1)

