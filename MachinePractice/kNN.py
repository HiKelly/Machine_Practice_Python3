from numpy import *     #导入科学计算包
import operator         #导入运算符模块

def createDataSet():    #创建数据集和标签
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):     #电影分类函数
    #inX是用于分类的输入向量
    #dataSet为输入的训练样本集
    #labels为标签向量
    #k为用于选择最近邻居的数目
    dataSetSize = dataSet.shape[0]  #dataSetSize表示样本集的个数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    #tile函数属于numpy，作用是让某个集合以某种方式重复，返回新的集合，详见博客https://www.jianshu.com/p/4b74a367833c
    #这里让inX这对坐标重复，以一维数组的形式重复样本集个数次，相减留下差值
    sqDiffMat = diffMat ** 2    #坐标的差的平方
    sqDistances = sqDiffMat.sum(axis=1) #坐标差的平方和，axis=0表示跨行，axis=1表示跨列，既每一项两个坐标的平方和
    distances = sqDistances ** 0.5  #坐标差的平方和开根号
    sortedDistIndicies = distances.argsort()    #对数据按照从小到大的次序排序
    classCount = {}     #统计前k个距离最小元素的分类字典
    for i in range(k):      #确定前k个距离最小元素所在的主要分类
        voteIlabel = labels[sortedDistIndicies[i]]  #第i个最小元素的标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  #给当前元素的标签出现的次数加1
    sortedClassCount = sorted(classCount.items(),
    key = operator.itemgetter(1), reverse = True)   
    #使用classCount.iteritems将字典拆解为元组列表，用operator.itemgetter(1)来获取元组列表第二列作为关键字排序，默认排序从大到小，所以reverse=True
    return sortedClassCount[0][0]


def file2matrix(filename):  #处理输入格式问题，输入文件名字符串，输出训练样本矩阵和类标签向量
    fr = open(filename) #打开文件
    arrayOLines = fr.readlines()    #逐行读入文件
    numberOfLines = len(arrayOLines)    #得到文件的行数
    returnMat = zeros((numberOfLines, 3))   #创建以0填充的矩阵，numberOfLines行，3列
    classLabelVector = []
    index = 0
    for line in arrayOLines:    #循环处理文件中的每行数据
        line = line.strip()     #截掉所有的回车字符
        listFromLine = line.split('\t') #将整行数据以'\t'分割成元素列表
        returnMat[index, :] = listFromLine[0:3]    #将前三个元素存储到特征矩阵中
        classLabelVector.append(int(listFromLine[-1]))  #索引值-1表示列表中的最后一列元素
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):  #归一化特征值，把数字特征值转化到0~1
    #归一化公式：newValue = (oldValue - min) / (max - min)
    minVals = dataSet.min(0)    #每列的最小值放到minVals中，参数0使得从每列中取
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet)) #用0填dataSet行数
    m = dataSet.shape[0]    #m是dataSet列数
    normDataSet = dataSet - tile(minVals, (m, 1))   #把minVals扩充到m个
    normDataSet = normDataSet / tile(ranges, (m, 1))    #在NumPy库中，矩阵除法需要使用函数 linalg.solve(matA,matB)
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt') #先处理数据格式
    normMat, ranges, minVals = autoNorm(datingDataMat)      #再归一化特征值
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)    #抽取10%作为测试数据
    errorCount = 0.0    #错误数
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],  #样本集和标签集都排除掉测试数据
            datingLabels[numTestVecs:m], 3)
        print("the classifier came back with {:2d}, the real answer is:{:2d}".format(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:    
            errorCount += 1.0
    print("the total error rate is:{:.3f}".format(errorCount/float(numTestVecs)))
