import numpy as np

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


xArr, yArr = loadDataSet('./machinelearninginaction/Ch08/ex0.txt')
ws = standRegres(xArr, yArr)
print(ws)