from math import log

def calcShannonEnt(dataSet):    #计算给定数据集的香农熵
    numEntries = len(dataSet)   #计算数据集中实例的总数
    labelCounts = {}    #每个键值记录当前类别出现的次数
    for featVec in dataSet:
        currentLabel = featVec[-1]  #当前标签是数据的最后一列
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0    #熵值
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries #该分类的概率
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value): #按照给定特征划分数据集，axis为划分数据集的特征，value为需要返回的特征的值
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            #把除了axis的特征添加到新的集合中
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:]) #把整个作为n项添加到后面
            retDataSet.append(reducedFeatVec)   #把整个作为一项添加到后面
    return retDataSet

def chooseBestFeatureToSplit(dataSet):  #选择最好的数据集划分方式
    numFeatures = len(dataSet[0]) - 1   #特征数量
    baseEntropy = calcShannonEnt(dataSet)   #基础香农熵
    bestInfoGain = 0.0  #最好的信息增益
    bestFeature = -1    #得到最好的信息增益的划分特征
    for i in range(numFeatures):    #计算每种划分方式的信息熵
        featList = [example[i] for example in dataSet]  #第i个特征有featList这些值
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:    #计算信息熵
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain: #更新最好的信息增益
            bestInfoGain = infoGain
            bestFeature = i
            print("here")
    return bestFeature