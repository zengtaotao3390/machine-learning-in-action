from numpy import *
import re
import random

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        # 求两个集合的并集
        vocabSet = set(document) | vocabSet
    return list(vocabSet)


def setOfWords2Vec(vocablarySet, inputSet):
    returnVec = [0] * len(vocablarySet)
    for word in inputSet:
        if word in vocablarySet:
            returnVec[vocablarySet.index(word)] = 1
        else:
            print('the word {} is not in my Vocabulary!'.format(word))
    return returnVec


def bagOfWords2Vec(vovablarySet, inputSet):
    returnVec = [0] * len(vovablarySet)
    for word in inputSet:
        if word in vovablarySet:
            returnVec[vovablarySet.index[word]] = 1
        else:
            print('the word {} is not in my Vocabulary!'.format(word))
    return returnVec

# listPosts, listClasses = loadDataSet()
# myVocabList = createVocabList(listPosts)
# print(myVocabList)
# print(setOfWords2Vec(myVocabList, listPosts[0]))
# print(setOfWords2Vec(myVocabList, listPosts[1]))


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p0Vect = p0Num / p0Denom
    p1Vect = p1Num / p1Denom
    return p0Vect, p1Vect, pAbusive

# listPosts, listClasses = loadDataSet()
# myVocabList = createVocabList(listPosts)
# trainMat = []
# for postinDoc in listPosts:
#     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
# print(myVocabList)
# p0Vect, p1Vect, pAbusive = trainNB0(trainMat, listClasses)
# print(pAbusive)
# print(p0Vect)
# print(p1Vect)


def classifyNB(vec2Classify, p0Vect, p1Vect, p1Class):
    p1 = sum(log(p1Vect) * vec2Classify) + log(p1Class)
    p0 = sum(log(p0Vect) * vec2Classify) + log(1 - p1Class)
    if p1 > p0 :
        return 1
    else:
        return 0


def testingNB():
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainMat = []
    for postinDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0Vect, p1Vect, pAbusive = trainNB0(trainMat, listClasses)
    testEntity = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntity))
    print('{}, classified as: {}'.format(testEntity, classifyNB(thisDoc, p0Vect, p1Vect, pAbusive)))
    testEntity = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntity))
    print('{}, classified as: {}'.format(testEntity, classifyNB(thisDoc, p0Vect, p1Vect, pAbusive)))

# testingNB()


# regEx = re.compile(r'\W*')
# emailText = open('./machinelearninginaction/Ch04/email/ham/6.txt').read()
# listOfTokens = regEx.split(emailText)
# listOfTokens = [token.lower() for token in listOfTokens if len(token) > 0]
# print(listOfTokens)


def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [token.lower() for token in listOfTokens if len(token) > 2]


def spamTest():
    fullText = []; trainText = []; classify = []
    for i in range(1, 26):
        with open('./machinelearninginaction/Ch04/email/spam/{}.txt'.format(i)) as file:
            emailText = file.read()
            listOfTokens = textParse(emailText)
            fullText.extend(listOfTokens)
            trainText.append(listOfTokens)
            classify.extend(1)
        with open('./machinelearninginaction/Ch04/email/ham/{}.txt'.format(i)) as file:
            emailText = file.read()
            listOfTokens = textParse(emailText)
            fullText.extend(listOfTokens)
            trainText.append(listOfTokens)
            classify.extend(0)
    testSet = []
    trainSet = range(50)
    # 原来的数据都被删除，所以随机数是一样的，值也是不一样的
    for i in range(10):
        randomIndex = int(random.uniform(len(trainSet)))
        testSet.append(trainSet[randomIndex])
        del(trainSet[randomIndex])
    # 开始训练
    trainMatrix = []
    trainClass = []
    vocabList = createVocabList(trainText)
    for docIndex in trainSet:
        trainMatrix.append(setOfWords2Vec(vocabList, trainText.index(docIndex)))
        trainClass.append(classify.index(docIndex))
    p0Vect, p1Vect, pAbusive = trainNB0(trainMatrix, classify)
    testMatrix = []
    testClass = []
    for docIndex in testSet:
        testMatrix.append(setOfWords2Vec(vocabList, trainText.index(docIndex)))
        testClass.append(classify.index(docIndex))

    for i in len(trainMatrix):
        testDoc = trainMatrix[i]
        pridictionClass = classifyNB(testDoc, p0Vect, p1Vect, testClass)
        errorCount = 0
        if pridictionClass != testClass[i]:
            errorCount += 1
        else:
            print('pridiction error text \n: {}', textParse())
    print('pridiction error late: {}', float(errorCount / 10))

spamTest()