from numpy import *
import operator
from os import listdir
from scipy import *

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def class0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diff = tile(inX,(dataSetSize,1))-dataSet
    squareDiff = diff**2
    distances = squareDiff.sum(axis=1)
    sqrt = distances**0.5
    sortedDistanceindex = sqrt.argsort()
    classCount={}
    for i in range(k):
        label = labels[sortedDistanceindex[i]]
        classCount[label]=classCount.get(label,1)+1
        sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def fileToMatrix(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    numberOfLines = len(arrayLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for i in range(len(arrayLines)):
        line = arrayLines[i]
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[i,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
    # print returnMat
    # print classLabelVector
    return returnMat,classLabelVector

# def plot():
#
#     datingDataMat,labels = fileToMatrix('datingTestSet.txt')


def autoNorm(dataSet):
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    ranges = maxVal-minVal
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVal,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVal

def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = fileToMatrix('datingTestSet.txt')
    normDataSet,ranges,minVal = autoNorm(datingDataMat)
    m = normDataSet.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = class0(normDataSet[i,:],normDataSet[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %s , the real answer is :%s" % (classifierResult,datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount+=1.0
    print "the total error size rate is: %f" % (errorCount/float(numTestVecs))

def imgTovector(filename):
    returnVector = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineString = fr.readline()
        for j in range(32):
            returnVector[0,i*32+j]=int(lineString[j])
    return returnVector

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileName = trainingFileList[i]
        fileStr = fileName.split('.')[0]
        classNumber = int(fileStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:] = imgTovector('digits/trainingDigits/%s' % fileName)
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileName = testFileList[i]
        fileStr = fileName.split('.')[0]
        classNumber = int(fileStr.split('_')[0])
        vectorUnderTest = imgTovector('digits/testDigits/%s' % fileName)
        classifierResult = class0(vectorUnderTest,trainingMat,hwLabels,3)
        print "the classifier came back with: %d, the real answer is :%d" % (classifierResult,classNumber)
        if (classifierResult != classNumber):
            errorCount+=1.0
    print "\ntotal error number is: %d" % errorCount
    print "\nthe error rate is: %f" % (errorCount/float(mTest))

if __name__ == '__main__':
    group,labels = createDataSet()
    print class0([0.9,0.5],group,labels,4)
    print group
    print labels
    datingDataMat,labels = fileToMatrix('datingTestSet.txt')
    normDataSet,ranges,minVal = autoNorm(datingDataMat)
    print ranges
    print minVal
    print normDataSet
    # datingClassTest()
    returnVector = imgTovector('digits/trainingDigits/0_13.txt')
    # print returnVector[0,0:31]
    handwritingClassTest()
