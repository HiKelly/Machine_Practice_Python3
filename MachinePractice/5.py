import logRegres

"""
dataArr, labelMat = logRegres.loadDataSet()
weights = logRegres.gradAscent(dataArr, labelMat)

from numpy import *
logRegres.plotBestFit(weights)  
"""

from numpy import *
dataArr, labelMat = logRegres.loadDataSet()
#weights = logRegres.stocGradAscent0(array(dataArr), labelMat)
weights = logRegres.stocGradAscent1(array(dataArr), labelMat)
logRegres.plotBestFit(weights)