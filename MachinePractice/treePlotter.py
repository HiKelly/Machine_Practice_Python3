import matplotlib.pyplot as plt

#python字体的默认设置没有中文字体，手动添加中文字体的名称
from pylab import *
plt.rcParams['font.sans-serif'] = ['SimHei']    #黑体

decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")  #决策节点，boxstyle为边框，sawtooth为锯齿形，fc是颜色深度
leafNode = dict(boxstyle = "round4", fc = "0.8")    #叶节点
arrow_args = dict(arrowstyle="<-")  #箭头格式

def plotNode(nodeTxt, centerPt, parentPt, nodeType):    #绘制带箭头的注解，nodeTxt节点文字，centerPt起点位置，parentPt终点位置，nodeType箭头类型
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

"""
annotate用于在图形上给数据添加文本注解
plt.annotate()的详细参数可用__doc__查看，如：print(plt.annotate.__doc__)
详见博客：https://blog.csdn.net/leaf_zizi/article/details/82886755
nodeTxt:注释文本的内容 xy:被注释的坐标点  xycoords:被注释点的坐标系属性 axes fraction:以子绘图区左下角为参考，单位是百分比
xytext：注释文本的坐标的 textcoords:注释文本的坐标系属性
va ha表示注释的坐标以注释框的正中心为准，而不是注释框的左下角
bbox是注释框的风格和颜色深度，fc越小，注释框的颜色越深
"""

def createPlot(inTree):   #绘制图像
    fig = plt.figure(1, facecolor='white')  #绘制背景为白色的图形
    fig.clf()   #清除原有图形
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #建立子图
    #createPlot.ax1为全局变量，绘制图像的句柄，subplot定义了一个绘图
    #111表示figure中的图有1行1列，第一个图
    #frameon表示是否绘制坐标轴矩形
    plotTree.totalW = float(getNumLeafs(inTree))    #树的宽是叶子个数
    plotTree.totalD = float(getTreeDepth(inTree))   #树的高度是深度
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0 #当前坐标在最上面
    plotTree(inTree, (0.5, 1.0), '')
    #createPlot.ax1 = plt.subplot(111,frameon=False)
    #plotNode(U'决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    #plotNode(U'叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

def getNumLeafs(myTree):    #获取叶节点的数目
    numLeafs = 0
    firstStr = list(myTree.keys())[0]     #dict.keys()返回一个字典所有的键
    secondDict = myTree[firstStr]   #第一个关键字是第一次划分数据集的类别标签
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':  #使用type()判断子节点是否为字典类型
            numLeafs += getNumLeafs(secondDict[key])    #递归调用函数，获取叶节点数目
        else:   numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):   #获取树的层数
    maxDepth = 0
    firstStr = list(myTree.keys())[0]   #python3变化，要先转化为list才能取第0项
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth:    maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):    #为了节省时间，预先存储树信息
    listOfTrees = [{'no surfacing':{0:'no', 1:{'flippers':{0:'no', 1:'yes'}}}},
                    {'no surfacing':{0:'no', 1:{'flippers':{0:{'head':{0:'no', 1:'yes'}}, 1:'no'}}}}]
    return listOfTrees[i]

def plotMidText(cntrPt, parentPt, txtString):   #计算tree中间位置 cntrPt起始位置，parentPt终止位置,txtString文本标签信息
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)  #把标签放在中间位置

def plotTree(myTree, parentPt, nodeTxt):
    #xOff和yOff记录当前要画的叶子节点的位置
    numLeafs = getNumLeafs(myTree)  #得到叶子数量
    depth = getTreeDepth(myTree)    #得到树的深度
    firstStr = list(myTree.keys())[0]   #得到树的第一个节点
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)   #计算子节点的坐标
    plotMidText(cntrPt, parentPt, nodeTxt)  #绘制线上的文字
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  #绘制节点
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD   #每绘制一次图，将y的坐标减少1.0/plottree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
