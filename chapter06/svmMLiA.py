#coding:utf-8
from numpy import *

def loadData(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j = i
    while(j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

    # mat将数组转换成矩阵
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMatrix = mat(classLabels).transpose()
    b = 0
    m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0
    while(iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            #计算支持向量机算法的预测值
            fXi = float(multiply(alphas,labelMatrix).T*(dataMatrix*dataMatrix[i,:].T))+b
            Ei = fXi - float(labelMatrix[i])
            if((labelMatrix[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMatrix[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                # multiply(alphas,labelMatrix) 是点乘    1*100   100*2  *   1*2
                fXj = float(multiply(alphas,labelMatrix).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMatrix[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMatrix[i] != labelMatrix[j]):
                    L = max(0,alphas[j]-alphas[i])
                    H = min(C,C+alphas[j]-alphas[i])
                else:
                    L = max(0,alphas[i]+alphas[j]-C)
                    H = min(C,alphas[j]+alphas[i])
                if L == H:
                    print "L==H"
                    continue
                # 根据公式计算未经剪辑的alphaj
                eta = 2.0*dataMatrix[i,:]*dataMatrix[j,:].T -dataMatrix[i,:]*dataMatrix[i,:].T -dataMatrix[j,:]*dataMatrix[j,:].T
                # 如果eta>=0,跳出本次循环
                if eta >= 0:
                    print "eta>=0"
                    continue
                alphas[j] -= labelMatrix[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                # 如果改变后的alphaj值变化不大，跳出本次循环
                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print "j not moving enough"
                    continue
                #否则，计算相应的alphai值
                alphas[i] += labelMatrix[j]*labelMatrix[i]*(alphaJold-alphas[j])
                b1 = b - Ei - labelMatrix[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMatrix[j]*(alphas[j] - alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMatrix[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMatrix[j]*(alphas[j] - alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if(0<alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1+b2)/2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print "iteration number: %d" % iter
    return b,alphas

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, Ktup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))
        # self.K = mat(zeros((self.m,self.m)))
        # for i in range(self.m):
        #     self.K[:,i] =

def calcEk(oS,k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T) + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS,k)
            deltaE = abs(Ei-Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK,Ej
    else:
        j = selectJrand(i,oS.m)
        Ej = calcEk(oS,j)
        return j,Ej

def updateEk(oS,k):
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]

def innerL(i, oS):
    Ei = calcEk(oS,i)
    if ((oS.labelMat[i]*Ei)<-oS.tol) and (oS.alphas[i] < oS.C) or ((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0,oS.alphas[j] - oS.alphas[i])
            H = min(oS.C,oS.C+oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0,oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C,oS.alphas[j] + oS.alphas[i])
        if L == H:
            print "L==H"
            return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0:
            print "eta>=0"
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print "j not moving enough"
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i)
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while(iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet: #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
                iter+=1
        else: #go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
                iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print "iteration number: %d" % iter
    return oS.b,oS.alphas

def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

if __name__ == '__main__':
    dataArr,labelArr = loadData('testSet.txt')
    print labelArr
    print dataArr
    # b,alphas = smoSimple(dataArr,labelArr,0.6,0.001,40)
    b,alphas = smoP(dataArr,labelArr,0.6,0.001,40)
    for i in range(100):
        if alphas[i]>0.0:
            print dataArr[i], labelArr[i]

    ws = calcWs(alphas,dataArr,labelArr)
    print ws
    print mat(dataArr)[0]*mat(ws) + b, labelArr[0]





