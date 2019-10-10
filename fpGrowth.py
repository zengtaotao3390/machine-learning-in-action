class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def increase(self, numOccur):
        self.count += numOccur


    def display(self, index=1):
        print(' '*index, self.name, '  ', self.count)
        for child in self.children.values():
            child.display(index + 1)

# rootNode = treeNode('pyramid', 9, None)
# rootNode.children['eye'] = treeNode('eye', 13, None)
# rootNode.display()
# rootNode.children['ophoenix'] = treeNode('phoenix', 3, None)
# rootNode.display()

# dataSet是一个字典，key是set，value为1
def createTree(dataSet, minSupport):
    # 头指针表，指向相同元素的节点，记录元素的个数
    headerTable = {}
    # 统计每个元素出现的个数
    for transation in dataSet:
        for item in transation:
            headerTable[item] = headerTable.get(item, 0) + dataSet[transation]
    # 删除不满足最小支持度的元素
    for key in list(headerTable):
        if headerTable[key] < minSupport:
           del(headerTable[key])
    frequentItemSet = set(headerTable.keys())
    # 没有符合条件的元素，直接退出
    if len(frequentItemSet) == 0:
        return None, None
    #创建头指针表指针
    for key in headerTable:
        headerTable[key] = [headerTable[key], None]
    # 创建根节点
    returnTree = treeNode('Null Set', 1, None)
    for transationSet, count in dataSet.items():
        # 当前事务数据每个元素的值，用于排序
        localD = {}
        for item in transationSet:
            if item in frequentItemSet:
                localD[item] = headerTable[item][0]
        # 超过一个元素排序
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p : p[1], reverse=True)]
            updateTree(orderedItems, returnTree, headerTable, count)
    return returnTree, headerTable


# 对排序的元素进行处理，加入树结构
def updateTree(items, inTree, headerTable, count):
    # 如果当前元素存在于树的子节点，直接增加元素对应子节点的计数
    if items[0] in inTree.children:
        inTree.children[items[0]].increase(count)
    else:
        # 加入子节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 更新头指针表指针
        if headerTable[items[0]][1] is None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    # 处理接下来的元素
    if len(items) > 1:
        updateTree(items[1:], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToLink, tagertNode):
    while(nodeToLink.nodeLink != None):
        nodeToLink = nodeToLink.nodeLink
    nodeToLink.nodeLink = tagertNode



def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    returnDic = {}
    for transaction in dataSet:
        returnDic[frozenset(transaction)] = 1
    return returnDic


simpleDat = loadSimpDat()
initSet = createInitSet(simpleDat)
myFPTree, myHeaderTab = createTree(initSet, 3)
myFPTree.display()
