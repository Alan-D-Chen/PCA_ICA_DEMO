# -*- coding: utf-8 -*-
# @Time    : 2019/10/19 下午 9:26
# @Author  : Alan D. Chen
# @FileName: ICA.py
# @Software: PyCharm
##################this is a simple for FastICA#################
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import time
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
#####################原始的数据和图片表示#############
C = 200  # 样本数S
x = np.arange(C)
s1 = 2 * np.sin(0.02 * np.pi * x)  # 正弦信号
a = np.linspace(-2, 2, 25)
s2 = np.array([a, a, a, a, a, a, a, a]).reshape(200,)  # 锯齿信号
s3 = np.array(20 * (5 * [2] + 5 * [-2]))  # 方波信号
s4 = 4 * (np.random.random([1, C]) - 0.5).reshape(200,)  # 随机信号
# drow origin signal
ax1 = plt.subplot(411)
ax2 = plt.subplot(412)
ax3 = plt.subplot(413)
ax4 = plt.subplot(414)

ax1.plot(x, s1)
ax1.spines["right"].set_color("y")
ax1.spines["top"].set_color("r")
ax1.spines["bottom"].set_position(("data", 0))
ax1.spines["left"].set_position(("data", 0))
ax1.set_title(now + "-orignal line 1")

ax2.plot(x, s2)
ax2.spines["right"].set_color("b")
ax2.spines["top"].set_color("g")
ax2.spines["bottom"].set_position(("data", 0))
ax2.spines["left"].set_position(("data", 0))
ax2.set_title(now + "-orignal line 2")

ax3.plot(x, s3)
ax3.spines["right"].set_color("y")
ax3.spines["top"].set_color("g")
ax3.spines["bottom"].set_position(("data", 0))
ax3.spines["left"].set_position(("data", 0))
ax3.set_title(now + "-orignal line 3")

ax4.plot(x, s4)
ax4.spines["right"].set_color("b")
ax4.spines["top"].set_color("r")
ax4.spines["bottom"].set_position(("data", 0))
ax4.spines["left"].set_position(("data", 0))
ax4.set_title(now + "-orignal line 4")
plt.savefig('/home/alanchen/Desktop/pydata/orignal_lines.jpg')
plt.show()
########################################################################
###################混合原始信号#########################
s = np.array([s1, s2, s3, s4])  # 合成信号
#print(s)
ran1 = np.random.random([4, 4])  # 随机矩阵
ran = 2 * ran1  # 随机矩阵
# print(ran)
# drow mix signal
mix = ran.dot(s)  # 混合信号（v）
ax1 = plt.subplot(411)
ax2 = plt.subplot(412)
ax3 = plt.subplot(413)
ax4 = plt.subplot(414)
ax1.plot(x, mix.T[:, 0])
ax1.set_title(now + "-mixed line 1")
ax2.plot(x, mix.T[:, 1])
ax2.set_title(now + "-mixed line 2")
ax3.plot(x, mix.T[:, 2])
ax3.set_title(now + "-mixed line 3")
ax4.plot(x, mix.T[:, 3])
ax4.set_title(now + "-mixed line 4")
plt.savefig('/home/alanchen/Desktop/pydata/mixed_lines.jpg')
plt.show()
####################################################################
##########################开始进行“FastICA”##########################
Maxcount = 10000  # %最大迭代次数
Critical = 0.00001  # %判断是否收敛
R, C = mix.shape
##################################数据中心化处理######################
average = np.mean(mix, axis=1)  # 计算行均值，axis=0，计算每一列的均值
for i in range(R):
    mix[i, :] = mix[i, :] - average[i]  # 数据标准化，均值为零

Cx = np.cov(mix)
value, eigvector = np.linalg.eig(Cx)  # 计算协方差阵的特征值和特征向量
print("\n此时获得的特征值为：\n", value, "\n此时获得特征向量为（横向）：\n", eigvector)
val = value ** (-1 / 2) * np.eye(R, dtype=float)
White = np.dot(val, eigvector.T)  # 白化矩阵
Z = np.dot(White, mix)  # 混合矩阵的主成分Z，Z为正交阵
print("\n此时获得的白化矩阵为：\n", White, "\n此时获得正交矩阵为：\n", Z)

##############################开始“牛顿迭代法”####################################
# W = np.random.random((R,R))# 4x4
W = 0.5 * np.ones([4, 4])  # 初始化权重矩阵
for n in range(R):
    count = 0  ####迭代次数####
    WP = W[:, n].reshape(R, 1)  # 初始化
    LastWP = np.zeros(R).reshape(R, 1)  # 列向量;LastWP=zeros(m,1);
    while LA.norm(WP - LastWP, 1) > Critical:  ####范数的运算#####
        print("\n运行次数为：", count, " ", now + "前后两者的迭代的差距\loop :", LA.norm(WP - LastWP, 1),)
        count = count + 1
        LastWP = np.copy(WP)  # %上次迭代的值
        gx = np.tanh(LastWP.T.dot(Z))  # 行向量

        for i in range(R):
            tm1 = np.mean(Z[i, :] * gx)
            tm2 = np.mean(1 - gx ** 2) * LastWP[i]  # 收敛快
            # tm2=np.mean(gx)*LastWP[i]     #收敛慢
            WP[i] = tm1 - tm2
            # print(" wp :", WP.T )
            WPP = np.zeros(R)  # 一维0向量
        for j in range(n):
            WPP = WPP + WP.T.dot(W[:, j]) * W[:, j]
            WP.shape = 1, R
            WP = WP - WPP
            WP.shape = R, 1
            WP = WP / (LA.norm(WP))  ###########更新wp################
            if count > Maxcount:
                print("Have reached Max count,exit loop:", LA.norm(WP - LastWP, 1))
                break
    print(now + "  loop count:", count)
    print("\n运行次数为：", count, " ", now + "前后两者的迭代的差距\loop :", LA.norm(WP - LastWP, 1), )
    W[:, n] = WP.reshape(R,)
SZ = W.T.dot(Z)
# plot extract signal
x = np.arange(0, C)

##########输出结果##############
ax1 = plt.subplot(411)
ax2 = plt.subplot(412)
ax3 = plt.subplot(413)
ax4 = plt.subplot(414)
ax1.plot(x, SZ.T[:, 0])
ax1.set_title(now + "-After FastICA line 1")
ax2.plot(x, SZ.T[:, 1])
ax2.set_title(now + "-After FastICA line 2")
ax3.plot(x, SZ.T[:, 2])
ax3.set_title(now + "-After FastICA line 3")
ax4.plot(x, SZ.T[:, 3])
ax4.set_title(now + "-After FastICA line 4")
plt.savefig('/home/alanchen/Desktop/pydata/After_lines.jpg')
plt.show()
