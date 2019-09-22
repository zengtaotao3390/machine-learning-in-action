import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    lines = []
    with open(fileName) as file:
        lines = file.readlines()
    numFeat = len(lines[0].split('\t')) - 1
    dataMat = []
    labelMat = []
    for line in lines:
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print('this matrix is singular, connot do inverse')
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


# xArr, yArr = loadDataSet('./machinelearninginaction/Ch08/ex0.txt')
# ws = standRegres(xArr, yArr)
# xMat = np.mat(xArr)
# yMat = np.mat(yArr)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
# xCopy = xMat.copy()
# # axis = 0,表示y轴
# xCopy.sort(0)
# print(xCopy)
# yHat = xCopy * ws
# ax.plot(xCopy[:, 1], yHat)
# plt.show()
# print(np.corrcoef((xMat * ws).T, yMat))


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


# print(yArr[0])
# print(lwlr(xArr[0], xArr, yArr, 1.0))
# print(lwlr(xArr[0], xArr, yArr, 0.001))
# print(np.mat(xMat))
# yHat = lwlrTest(xArr, xArr, yArr, 1.0)
# # xMat = np.mat(xArr)
# # srtInd = xMat[:, 1].argsort(0)
# # xSort = xMat[srtInd][:, 0, :]
# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # ax.plot(xSort[:, 1], yHat[srtInd])
# # ax.scatter(xMat[:, 1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2, c='red')
# # plt.show()

def rssError(yArr, yHarArr):
    return ((yArr - yHarArr)**2).sum()


# abX, abY = loadDataSet('./machinelearninginaction/Ch08/abalone.txt')
# yHat01 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 0.1)
# yHat1 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 1)
# yHat10 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 10)
# print(rssError(abY[0: 99], yHat01))
# print(rssError(abY[0: 99], yHat1))
# print(rssError(abY[0: 99], yHat10))


# yHat01 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 0.1)
# yHat1 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 1)
# yHat10 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 10)
# print(rssError(abY[100: 199], yHat01))
# print(rssError(abY[100: 199], yHat1))
# print(rssError(abY[100: 199], yHat10))
#
#
# ws = standRegres(abX[0: 99], abY[0: 99])
# yHat = np.mat(abX[100: 199]) * ws
# print(rssError(abY[100: 199], yHat.T.A))


def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print('this matrix is singular, cannot do inverse')
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wmat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wmat[i: ] = ws.T
    return wmat


# abX, abY = loadDataSet('./machinelearninginaction/Ch08/abalone.txt')
# redgeWeights = ridgeTest(abX, abY)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(redgeWeights)
# plt.show()


# 向前逐步线性回归
def regularize(xMat):
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    return (xMat - xMeans) / xVar

def stageWise(xArr, yArr, dis=0.01, numItera=100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numItera, n))
    ws = np.zeros((1, n))
    testWs = ws.copy()
    bestWs = ws.copy()
    minRssError = np.inf
    for i in range(numItera):
        print(ws)
        for j in range(n):
            for direction in [-1, 1]:
                testWs = ws.copy()
                testWs[:, j] += direction * dis
                prediction = xMat * testWs.T
                thisRssError = rssError(yMat.A, prediction.A)
                if thisRssError < minRssError:
                    minRssError = thisRssError
                    bestWs = testWs
        ws = bestWs.copy()
        returnMat[i, :] = ws
    return returnMat


abX, abY = loadDataSet('./machinelearninginaction/Ch08/abalone.txt')
stageWise(abX, abY, 0.01, 200)