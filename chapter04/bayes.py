#coding:utf-8
import feedparser
from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def trainNB(trainMatrix,trainCategory):
    numberTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numberTrainDocs)
    # p0Num = zeros(numWords)
    # p1Num = zeros(numWords)
    # p0Denom = 0.0
    # p1Denom = 0.0
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numberTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
            # print 'p1Num', p1Num
            # print 'p1Denom', p1Denom
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
            # print 'p0Num', p0Num
            # print 'p0Denom', p0Denom
    # p1Vect = p1Num/p1Denom
    # p0Vect = p0Num/p0Denom
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWordsVec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    # print 'vec2Classify',vec2Classify
    # print 'p1Vec',p1Vec
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def bagOfWord2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec


###  下面做垃圾邮件的demo

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [item.lower() for item in listOfTokens if len(item) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClassed = []
    for docIndex in trainingSet:
        trainMat.append(setOfWordsVec(vocabList,docList[docIndex]))
        trainClassed.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB(array(trainMat),array(trainClassed))
    errorCount = 0
    for docIndex in testSet:
        word2Vector = setOfWordsVec(vocabList,docList[docIndex])
        if classifyNB(word2Vector,p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    # 这个数字是随机的
    print 'the error rate is: ' , float(errorCount)/len(testSet)

def calcMostFred(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(),key = operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    docList=[]
    classList=[]
    fullText=[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFred(vocabList,fullText)
    # 去掉高频词汇
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = range(2*minLen)
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]
    trainClass=[]
    for docIndex in trainingSet:
        trainMat.append(bagOfWord2VecMN(vocabList,docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB(array(trainMat),array(trainClass))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWord2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ', float(errorCount)/len(testSet)
    return vocabList,p0V,p1V


if __name__ == '__main__':
    postingList,classVec = loadDataSet()
    myVocabList = createVocabList(postingList)
    print myVocabList
    trainMat = []
    for item in postingList:
        trainMat.append(setOfWordsVec(myVocabList,item))
    print trainMat
    p0Vect,p1Vect,pAbusive = trainNB(array(trainMat),array(classVec))
    # print p0Vect
    # print p1Vect
    # print pAbusive
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWordsVec(myVocabList,testEntry))
    print testEntry,'classified as:',classifyNB(thisDoc,p0Vect,p1Vect,pAbusive)
    #testEntry = ['stupied','garbage','love','my'] #0
    #testEntry = ['stupied','garbage','love'] #1
    #testEntry = ['stupied','garbage'] #1
    testEntry = ['stupied','garbage']
    thisDoc = array(setOfWordsVec(myVocabList,testEntry))
    print testEntry,'classified as:',classifyNB(thisDoc,p0Vect,p1Vect,pAbusive)
    spamTest()
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    vocabList,p0V,p1V = localWords(ny,sf)
    print vocabList
    print p0V
    print p1V
