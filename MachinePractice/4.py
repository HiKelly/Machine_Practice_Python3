import bayes
from numpy import *

"""
listOPosts, listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts) #创建单词集合
print(myVocabList)

print(bayes.setOfWords2Vec(myVocabList, listOPosts[0]))
print(bayes.setOfWords2Vec(myVocabList, listOPosts[3]))

listOPosts, listClasses = bayes.loadDataSet()   #从预先加载值中调入数据
myVocabList = bayes.createVocabList(listOPosts) #创建包含所有词的列表
trainMat = []
for postinDoc in listOPosts:    #将文档转化为数字
    trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
print(myVocabList)
print(pAb)
print(p0V)
print(p1V)
"""

#print(bayes.testingNB())
"""
import re
mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
regEx = re.compile('\\W')  #'\W'匹配非字母字符， '*'匹配前一个字符0或多次
listOfTokens = re.split(regEx, mySent)
#print([tok.lower() for tok in listOfTokens if len(tok) > 0])

emailText = open('email/ham/6.txt').read()
listOfTokens = regEx.split(emailText)
print(listOfTokens)
"""

print(bayes.spamTest())
print(bayes.spamTest())