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

# 最大最小之间的随机数
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.ones((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids


# dataMat = np.mat(loadDataSet('./machinelearninginaction/Ch10/testSet.txt'))
# print(min(dataMat[:, 0]))
# print(max(dataMat[:, 0]))
# print(min(dataMat[:, 1]))
# print(max(dataMat[:, 1]))
# print(randCent(dataMat, 2))
# print(disEclud(dataMat[0], dataMat[1]))


def kMeans(dataSet, k, distMeans=disEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeans(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        # print(centroids)
        for cent in range(k):
            ptsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A==cent)[0]]
            centroids[cent, :] = np.mean(ptsInCluster, axis=0)
    return centroids, clusterAssment





def biKmeans(dataSet, k, distMeans=disEclud):
    m = np.shape(dataSet)[0]
    # 第一个值对应样本所属的质心，第二个值对应样本到质心的距离
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 第一个质心
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    # 找到所有样本到初始质心的距离
    for j in range(m):
        clusterAssment[j, 1] = distMeans(np.mat(centroid0), dataSet[j, :]) ** 2
    while(len(centList) < k):
        # 初始化最小的误差为正无穷
        lowestSSE = np.inf
        for i in range(len(centList)):
            # 得到当前簇的样本
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A==i)[0], :]
#             使用K-均值算法对当前簇进行二分
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeans)
#             得到切分后的误差平方和
            sseSplit = np.sum(splitClustAss[:, 1])
#             计算没有切分簇的误差，切分后总误差=切分后簇的误差 + 没有切分簇的误差
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print('sseSplit, and notSplit: ', sseSplit, sseNotSplit)
            # 切分后的总误差小于切分前的总误差，那么将保留此切分
            if(sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
#             更新簇的分配结果
#       将二分聚类为1的赋值为总的聚类长度
#       将二分聚类为0的赋值父类的聚类index
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
#       添加质心
        centList[bestCentToSplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1, :])
#       更改样本的执行和距离
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return centList, clusterAssment


dataMat3 = np.mat(loadDataSet('./machinelearninginaction/Ch10/testSet2.txt'))
centList, myNewAssments = biKmeans(dataMat3, 3)
print(centList)





