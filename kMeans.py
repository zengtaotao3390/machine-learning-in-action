import numpy as np


def loadDataSet(fileName):
    dataMat = []
    lines = []
    with open(fileName) as fr:
        lines = fr.readlines()
    for line in lines:
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


def disEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.ones((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids


dataMat = np.mat(loadDataSet('./machinelearninginaction/Ch10/testSet.txt'))
# print(min(dataMat[:, 0]))
# print(max(dataMat[:, 0]))
# print(min(dataMat[:, 1]))
# print(max(dataMat[:, 1]))
print(randCent(dataMat, 2))
print(disEclud(dataMat[0], dataMat[1]))