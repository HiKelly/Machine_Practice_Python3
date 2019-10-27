import trees
from imp import reload

"""
reload(trees.py)   直接使用不用重新加载
myDat, labels = trees.createDataSet()
print(myDat)
print(trees.calcShannonEnt(myDat))

#增加一个'maybe'的分类，熵增加，混乱度增加
myDat[0][-1] = 'maybe' 
print(myDat)
print(trees.calcShannonEnt(myDat))

myDat, labels = trees.createDataSet()
print(myDat)
print(trees.splitDataSet(myDat, 0, 1))
print(trees.splitDataSet(myDat, 0, 0))


myDat, labels = trees.createDataSet()
print(trees.chooseBestFeatureToSplit(myDat))
print(myDat)

myDat, labels = trees.createDataSet()
myTree = trees.createTree(myDat, labels)
print(myTree)

import treePlotter
treePlotter.createPlot()

import treePlotter
print(treePlotter.retrieveTree(1))
myTree = treePlotter.retrieveTree(0)
print(treePlotter.getNumLeafs(myTree))
print(treePlotter.getTreeDepth(myTree))
"""

import treePlotter
myTree=treePlotter.retrieveTree(0)
myTree['no surfacing'][3] = 'maybe'
treePlotter.createPlot(myTree)