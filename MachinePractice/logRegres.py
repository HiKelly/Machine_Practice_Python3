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

def plotBestFit(wei):   #画出数据集和Logistic回归最佳拟合直线的函数
    import matplotlib.pyplot as plt
    #weights = wei.getA()    #getA()函数将矩阵转化为数组
    weights = wei
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [];    ycord1 = []
    xcord2 = [];    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);   ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]);   ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')   #类别1用红色标记
    ax.scatter(xcord2, ycord2, s=30, c='green')     #类别2用绿色标记
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]   #按照sigmoid的图像0为分界处，设0=w0x0 + w1x1 + w2x2 x0=1解x1和x2的关系
    ax.plot(x, y)
    plt.xlabel('X1');   plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):   #随机梯度上升算法
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))   #h为数值
        error = classLabels[i] - h  #error为数值  
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter = 150):    #改进的随机梯度上升算法
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):    
        dataIndex = list(range(m))    #默认迭代更新150词
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01    #alpha每次迭代时需要调整
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))   #随机选取样本更新回归系数，将减少周期性的波动
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
