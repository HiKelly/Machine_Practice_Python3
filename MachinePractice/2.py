import kNN
from numpy import *
from imp import reload

"""
电影分类
group, labels = kNN.createDataSet()
print(kNN.classify0([0, 0], group, labels, 3))

约会网站分类
from imp import reload
reload(kNN) #python3使用reload时需要先 from imp import reload
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')

import matplotlib
import matplotlib.pyplot as plt #引用matplotlib.pyplog可直接使用plt,pyplot是常用的画图模块
fig = plt.figure()
ax = fig.add_subplot(111)   #画子图，参数1：子图总行数；参数2：子图总列数；参数三：子图位置；返回坐标轴实例
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))    #使用scatter()函数绘制散点图，传递一对x和y坐标
plt.show()

reload(kNN)
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
print(normMat)
"""

reload(kNN)
kNN.datingClassTest()
