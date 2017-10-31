#coding:utf-8
from math import log
import operator

def calcShannonEnt(dataSet):
    numEntties = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel] += 1
    shannonEnt = 0
    for key in labelCounts:
        prob  = float(labelCounts[key])/numEntties
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

def intrinsicValue(fretureCounts,num):
    iv=0.0
    for key in fretureCounts.keys():
        prob = float(fretureCounts[key])/num
        iv -= prob*log(prob,num)
    return iv

def calcGain(dataSet):
    # 计算出有多少个特征
    numFeature = len(dataSet[0]) - 1
    # 每个特征需要统计有多少个D
    for i in range(numFeature):
        # 拿到第i个feature的列表
        print "拿到第%d个feature的列表" % i
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        fretureCounts = {}
        sumEntropy = 0.0
        num = len(featList)
        # print num
        for feature in featList:
            if feature not in fretureCounts.keys():
                fretureCounts[feature] = 0
            fretureCounts[feature] += 1
        for feature in uniqueVals:
            subDataSet = []
            subDataSet.extend([example for example in dataSet if example[i] == feature])
            print subDataSet
            # 计算每个子的Ent（D^v）
            sumEntropy += (int(fretureCounts[feature])/float(num))*calcShannonEnt(subDataSet)
        Gain = calcShannonEnt(dataSet) - sumEntropy
        iv = intrinsicValue(fretureCounts,num)
        print 'iv:',iv
        Gain_ratio = Gain/iv
        print Gain_ratio

def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,0,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0]) - 1
    baseEntropy  = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeature):
        print "按照第 %d 个特征划分" % i
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        print "特征划分",uniqueVals
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote] +=1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

if __name__ == '__main__':
    dataSet,labels = createDataSet()
    print calcShannonEnt(dataSet)
    print chooseBestFeatureToSplit(dataSet)
    print createTree(dataSet,labels)
    calcGain(dataSet)
