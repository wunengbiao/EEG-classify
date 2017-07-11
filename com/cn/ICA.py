# -*- coding: utf-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
from com.cn.CFEEG import read_data

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=10)
from sklearn.decomposition.fastica_ import FastICA
import copy

path = "E:\EEG\理想数据k=1~2\k=1.0\祥-调节2hz_20160620_102006_ASIC_EEG.xlsx"

dataSet = read_data(path, '第一组', row=(0,), col=(5, 13), target_col=21)
data = dataSet['data']
header = dataSet['header']
row, col = np.shape(data)

x = [i for i in range(row)]
fig = plt.figure(1)
fig2 = plt.figure(2)
# fig3=plt.figure(3)

fast_ica = FastICA(8, fun='exp', max_iter=1000)
new_data = fast_ica.fit_transform(data)

temp = new_data
# temp=temp*10
# print(np.linalg.norm(data,2,axis=0).argsort())
# change=np.linalg.norm(temp,2,axis=0)
# print(change.argsort())
#
#
#

ava_data = np.mean(data, axis=0)
order = []

for i in range(8):
    H = np.c_[temp, temp * (-1)]
    H = H + 2
    # temp=np.square(temp)
    H *= (ava_data[i] / 2)
    min = float('inf')
    index = -1
    for j in range(16):
        if j in order or (j - 8) in order: continue
        a = np.linalg.norm(data[:, i] - H[:, j], 1)
        if a < min:
            min = a
            index = j
    if index >= 8:
        index = index - 8
    order.append(index)

print(order)
ica_order = np.array(order).argsort()
print(ica_order)

ica_data = np.c_[new_data[:, ica_order[0]], new_data[:, ica_order[1]], new_data[:, ica_order[2]], new_data[:, ica_order[
    3]], new_data[:, ica_order[4]], new_data[:, ica_order[5]], new_data[:, ica_order[6]], new_data[:, ica_order[7]]]

# print(ica_data)

for i in range(8):
    y = data[:, i]
    axes = fig.add_subplot(8, 1, i + 1)
    axes.xaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度
    axes.yaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度
    axes.set_xlabel('t', fontproperties=font)
    axes.set_ylabel(header[i], fontproperties=font)
    axes.plot(x, y, 'r-')

    # y_ica = new_data[:, i]
    # axes_ica = fig2.add_subplot(8, 1, i + 1)
    # axes_ica.xaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度
    # axes_ica.yaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度
    # axes_ica.set_xlabel('t', fontproperties=font)
    # axes_ica.plot(x, y_ica, 'r-')
    #
    # y_ica = new_data[:, i]*(-1)
    # axes_ica = fig3.add_subplot(8, 1, i + 1)
    # axes_ica.xaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度
    # axes_ica.yaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度
    # axes_ica.set_xlabel('t', fontproperties=font)
    # axes_ica.plot(x, y_ica, 'r-')

    y = ica_data[:, i]
    axes = fig2.add_subplot(8, 1, i + 1)
    axes.xaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度
    axes.yaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度
    axes.set_xlabel('t', fontproperties=font)
    axes.set_ylabel(header[i], fontproperties=font)
    axes.plot(x, y, 'r-')

plt.show()
