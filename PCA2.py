# -*- coding: utf-8 -*-
# @Time    : 2019/10/16 上午 10:57
# @Author  : Alan D. Chen
# @FileName: PCA2.py
# @Software: PyCharm

###this is a simple example for principal component analysis.
import pandas as pd
import xlrd
import numpy as np

df=pd.read_excel('E:\pycharm-items-github\grade3.xls')#这个会直接默认读取到这个Excel的第一个表单
data=df.head(10)#默认读取前5行的数据
print("The result is :\n{0}".format(data))#格式化输出
##print(df.describe())##describe the data

"""
data1=df.ix[[1,2,3]].values#读取多行方面内容
print("The result is :\n{0}".format(data1))#
"""
"""
data2=df.ix[[1,2,5,10],[1,2,5,9]].values#读取第一行第二行的title以及data列的值，这里需要嵌套列表
print("读取指定行的数据：\n{0}".format(data2))
"""
"""
data2=df.ix[[1,2,5,10],['chinese','lizong']].values#读取第一行第二行的title以及data列的值，这里需要嵌套列表
print("读取指定行的数据：\n{0}".format(data2))
"""
"""
print("输出行号列表",df.index.values)
print("输出列标题",df.columns.values)
"""
#data3=df.ix[:,[2,3,4,5,6,7,8,9]].values#读取第一行第二行的title以及data列的值，这里需要嵌套列表;这里的data3已经是矩阵了。
#data3=df.ix[:,:].values
data3=df.ix[[2,3,4,5],[2,3,4,5]].values
#print("读取指定行列的数据：\n{0}".format(data3))
print("\n 输入矩阵为： \n",data3,"\n 数据类型：",type(data3),"\n 该矩阵的行列为",data3.shape[0],"*",data3.shape[1])
Z = np.zeros((data3.shape[0],data3.shape[1]))

for i in range(data3.shape[1]):###此处中心化数据
    Z[:,i]= data3[:,i] - np.mean(data3[:,i])
    #print(data3[:,i] - np.mean(data3[:,i]))
print("\n 数据中心化结果为: \n",Z,"\n 该矩阵的行列为",Z.shape[0],"*",Z.shape[1])

Z2 = np.cov(Z.T) ##特征协方差矩阵
print("\n 求得特征协方差矩阵结果为: \n",Z,"\n 该矩阵的行列为",Z.shape[0],"*",Z.shape[1])

eigenvalue,featurevector=np.linalg.eig(Z2) #计算特征值和特征向量
print("\n 特征值 入 /the eigenvalues ：\n",eigenvalue)
print(" 特征向量 /the featurevectors ：\n",featurevector)
####根据特征值的大小，将特征值和特征向量按照由大到小的顺序输出
print("\n 根据特征值的大小，将特征值和特征向量按照由大到小的顺序输出:")
l1 = []
l2 = []
for k in range(Z2.shape[0]):
    l1.append(eigenvalue[k])
    l2.append(featurevector.T[k])
#print("\n 特征值为  ：\n",l1,"\n 特征向量为：\n",l2)
count = len(l1)
for i in range(0, count):
    for j in range(i + 1, count):
        if l1[i] < l1[j]:
            l1[i], l1[j] = l1[j], l1[i]
            l2[i], l2[j] = l2[j], l2[i]
    print(l1[i],":--->",l2[i])
######在取得最大特征值和对应特征向量后，得到的结果：
print("\n 在取得最大特征值和对应特征向量后，得到的结果：")
Y = np.dot(Z,l2[0].reshape(-1,1))
print("dataAdjust \n",Z,"\n 最大特征值对应特征向量: \n",l2[0].reshape(-1,1))
print("结果为：\n",Y)