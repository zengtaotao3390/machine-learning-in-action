from numpy import *

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


listPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listPosts)
print(myVocabList)
print(setOfWords2Vec(myVocabList, listPosts[0]))
print(setOfWords2Vec(myVocabList, listPosts[1]))


# sum1 = sum([1, 1, 0, 1])
# print(sum1)
print(sum(ones(3)))
