def loadDataSet():
    # return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    return  [['l1', 'l2', 'l5'], ['l2', 'l4'], ['l2', 'l3'],
            ['l1', 'l2', 'l4'], ['l1', 'l3'], ['l2', 'l3'],
            ['l1', 'l3'], ['l1', 'l2', 'l3', 'l5'], ['l1', 'l2', 'l3']]

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
            # 当k-2项相同时，两个集合合并：例如k= 3 {0,1}{0,2}{1,2} 最终结果为{0,1,2}
            # 当k=4 有{0,1,3}{0,1,2}{1,2,3} {1,2,4} 最终合并结果为4的集合有{0,1,2,3}{1,2,3,4}
            L1 = list(Lk[i])[: k-2]
            L2 = list(Lk[j])[: k-2]
            L1.sort()
            L2.sort()
            # 对应元素相同才能相等
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


# 寻找频繁项集
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


# L, supportData = apriori(dataSet, 0.5)
# print(L)
# print(L[0])
# print(L[1])
# print(L[2])
# print(L[3])


def generateRules(L, supportData, minConfidence=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for frequentSet in L[i]:
            # 生成只有一个元素的集合列表
            H1 = [frozenset([item]) for item in frequentSet]
            if(i > 1):
                # 发现原书对于大于1元素的集合，没有进行置信度计算，一下是自己新加的
                Hmp1 = calculateConfidence(frequentSet, H1, supportData, bigRuleList, minConfidence)
                # 对有效子集进行合并运算
                if(len(Hmp1) > 1):
                    relusFromConsequence(frequentSet, H1, supportData, bigRuleList, minConfidence)
            else:
                calculateConfidence(frequentSet, H1, supportData, bigRuleList, minConfidence)


def calculateConfidence(frequentSet, H, supportData, bigRuleList, minConfidence=0.7):
    prunedH = []
    for consequence in H:
        confidence = supportData[frequentSet] / supportData[frequentSet - consequence]
        if confidence >= minConfidence:
            print(frequentSet - consequence, '---->', consequence, 'confidence: ', confidence)
            bigRuleList.append((frequentSet - consequence, consequence, confidence))
            prunedH.append(consequence)
    return prunedH


def relusFromConsequence(frequestSet, H, supportData, BigRuleList, minConfidence=0.7):
    m = len(H[0])
    if len(frequestSet) > (m + 1):
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calculateConfidence(frequestSet, Hmp1, supportData, BigRuleList, minConfidence)
        if(len(Hmp1) > 1):
            relusFromConsequence(frequestSet, Hmp1, supportData, BigRuleList, minConfidence)

L, supportData = apriori(dataSet, 0.2)
generateRules(L, supportData, 0.7)

# mushDataSet = [line.split() for line in open('./machinelearninginaction/Ch11/mushroom.dat').readlines()]
# L, suppData = apriori(mushDataSet, minSupport=0.3)
# for item in L[3]:
#     if item.intersection('2'):
#         print(item)