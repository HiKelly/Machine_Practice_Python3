from numpy import *

def loadDataSet():  #创建实验样本
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],    #进行词条切分后的文档集合
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]   #类别标签集合 1代表侮辱性文字，0代表正常言论
    return postingList, classVec

def createVocabList(dataSet):   #创建一个包含在所有文档中出现的不重复词的列表
    vocabSet = set([])  #先创建一个空集合
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

#词集模型
def setOfWords2Vec(vocabList, inputSet):    #判断词汇表中的单词在输入文档中是否出现
    returnVec = [0] * len(vocabList)    #创建一个和词汇表等长的向量，将其元素都设置为0
    for word in inputSet:  #遍历单词表
        if word in vocabList:   #出现了词汇表中的单词，文档向量对应值设为1
            returnVec[vocabList.index(word)] = 1
        else:   print("the word: %s is not in my Vocabulary!" % word)
    return returnVec    #输出文档向量，向量的每一元素为1或0，表示词汇表中的单词在输入文档中是否出现

#词袋模型
def bagOfWords2VecMN(vocabList, inputSet):  #上面函数的改良版，计算权重
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB0(trainMatrix, trainCategory):   #朴素贝叶斯分类器训练函数
    #trainMatrix为文档矩阵，trainCategory为每篇文档类别标签所构成的向量
    numTrainDocs = len(trainMatrix) #文档个数
    numWords = len(trainMatrix[0])  #每篇文档的长度
    pAbusive = sum(trainCategory) / float(numTrainDocs) #侮辱性文档的概率
    #进行文档分类时，要计算多个概率的乘积，如果其中一个概率为0，最后的乘积也为0
    #为降低影响，将所有词的出现数初始化为1，将分母初始化为2
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):   #遍历训练文档
        if trainCategory[i] == 1:   #这是一篇侮辱性文档
            p1Num += trainMatrix[i] #统计侮辱性文档中各个词条出现的次数
            p1Denom += sum(trainMatrix[i])  #统计所有侮辱性文档中出现的单词数目
        else:   #这是一篇正常文档
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)    #利用Numpy数组计算p(wi|c1)    为避免下溢出问题，改为log()
    p0Vect = log(p0Num / p0Denom)    #利用Numpy数组计算p(wi|c0)    为避免下溢出问题，改为log()
    #下溢出问题：太多很小的数相乘造成，四舍五入会得到0
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)   #log(a*b) = log(a) + log(b)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:    
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc =array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))