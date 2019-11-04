from numpy import *

def loadDataSet():  #打开文件testSet.txt并逐行读取
    dataMat = [];   labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines(): #逐行读取文件
        lineArr = line.strip().split()  #先去掉每行开头结尾的符号，再按空格分开
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #将x0设为1.0
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat    #返回数据矩阵与标签矩阵

def sigmoid(inX):   #海维赛德阶跃函数（单位阶跃函数）   类似于阶跃函数
    return 1.0 / (1 + exp(-inX))    #a(z) = 1 / (1 + e^(-z))

def gradAscent(dataMatIn, classLabels): #梯度上升算法
    dataMatrix = mat(dataMatIn) #转换为NumPy矩阵数据类型
    labelMat = mat(classLabels).transpose() #矩阵转置，为了便于矩阵运算
    m,n = shape(dataMatrix) #dataMatrix为m行n列
    alpha = 0.001   #向目标移动的步长
    maxCycles = 500 #迭代次数
    weights = ones((n, 1))  #把n*1的2维数组填充为1
    for k in range(maxCycles):  #皆为矩阵运算
        h = sigmoid(dataMatrix * weights)   #h为列向量 m*n * n*1 = m*1
        error = (labelMat - h)  #计算真实类别与预测类别的差值，略去了数学推导
        weights = weights + alpha * dataMatrix.transpose() * error  #按差值方向调整回归系数
    return weights  #返回训练好的回归系数
