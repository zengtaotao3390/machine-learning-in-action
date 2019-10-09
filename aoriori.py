def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


# Ck候选项集列表  返回支持项
# return 满足条件的集合和所有集合的支持度
def scanD(D: list, Ck: list, minSupport):
    ssCnt = { }
    for tid in D:
       for can in Ck:
           if can.issubset(tid):
               if can not in ssCnt:
                   ssCnt[can] = 1
               else:
                   ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


dataSet = loadDataSet()
# C1 = createC1(dataSet)


# 创建候选集  就是父集
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            # 当k-2项相同时，两个集合合并：例如k= 3 {0,1}{0,2}{1,2} 最终结果为{1,2,3}
            # 当k=4 有{0,1,3}{0,1,2}{1,2,3} {1,2,4} 最终合并结果为4的集合有{0,1,2,3}{1,2,3,4}
            L1 = list(Lk[i])[: k-2]
            L2 = list(Lk[j])[: k-2]
            L1.sort()
            L2.sort()
            # 对应元素相同才能相等
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    # 初始合并为元素个数为2的集合
    k = 2
    while len(L[k - 2]) > 0:
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


L, supportData = apriori(dataSet, 0.7)
print(L)
print(L[1])
print(L[2])
# print(L[3])

