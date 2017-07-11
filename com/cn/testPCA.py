from com.cn.CFEEG import read_data,mergeFile
from com.cn.CFEEG import pca,groupByKByName,mergeData,readFTM,read_data,eigen
import numpy as np
import weakref
import matplotlib.pyplot as plt
# path="E:\GitRepository\EEG\Data\全部数据\原始excel\k=1.0\wnb-201608281115_20160829_111545_ASIC_EEG.xlsx"

# pca(3,path,'第一组',row=(0,),col=(5,13),target_col=21)

# result=groupByKByName('E:\EEG\k')
# print(result)

# resultDict=mergeData("E:\\EEG\k=1.3",'第一组', (0,), (13, 21), 21,10,3)
# print(resultDict['data'])

# import os
# path="E:\\EEG\\k\\k=1.1"
#
# paths=os.listdir(path)
#
# new_path=[os.path.join(path,i) for i in paths]


# value,vector=eigen(new_path)
# print(value)
# print(vector)

#
# result=read_data("E:\EEG\k\k=1.0\cxq-早上2hz_20160601_103622_ASIC_EEG.xlsx",'第一组',row=(0,),col=(5,13),target_col=21)
# print(result)

# for p in new_path:
#     result=read_data(p,'第一组',row=(0,),col=(5,13),target_col=21)
#     print(result)


# a=[1,3,5,7]
# print(a.remove(a[1]))
# print(a)
# a=[1,2,3]
# a=a+[0 for i in range(5)]+[1 for j in range(5)]
# print(a)
# import copy
# a=[1,2,3,4,5]
#
# for i in range(len(a)):
#     temp=copy.copy(a)
#     temp.remove(temp[i])
#     print(temp)

# filename="E:\EEG\k\k=1.0\cxq-早上2hz_20160601_103622_ASIC_EEG.xlsx"
#
# print(os.path.dirname(filename))
# print(os.path.basename(filename))

# from numpy.linalg import eig
#
# path="E:\EEG\k\k=1.0"
#
# dataSet=mergeFile(path,'第一组',row=(0,),col=(5,13),target_col=21,window_size=10,step=3)
# data=dataSet['data']
# row,col=np.shape(data)
# data2=np.cov(data.T)
#
# diffMat=np.mean(data,axis=0)
#
# data=data-np.tile(diffMat,[row,1])
#
# print(np.shape(data))
#
# data=np.dot(data.T,data)/(row-1)
#
#
#
# eigenvalue,eigenvector=eig(data)
# eigenvalue2,eigenvector2=eig(data2)
#
# print(eigenvector)
# print(eigenvector2)
# accuracy_GBC=[i for i in range(11)]
# accuracy_GBR=[i for i in range(11,22)]
# accuracy_RFC=[i for i in range(22,33)]

# fig=plt.figure()
# axes=fig.add_subplot(111)
#
# x=np.linspace(1.0,2.0,11)
#
# axes.set_xlabel('K')
# axes.set_ylabel('Accuracy Rate')
# plt.plot(x,accuracy_GBC,'r-',label='GBC')
# plt.plot(x,accuracy_GBR,'b-',label='GBR')
# plt.plot(x,accuracy_RFC,'g-',label='RCR')
# legend = plt.legend(loc='upper left', shadow=True, fontsize='x-small')
# legend.get_frame().set_facecolor('#00FFCC')
# plt.show()
# fig=plt.figure()
# axes=fig.add_subplot(111)
# #
# x=np.linspace(1.0,2.0,11)
#
# fig.suptitle("PCA Process Data")
# axes.set_xlabel('K')
# axes.set_ylabel('Accuracy Rate')
# plt.plot(x,accuracy_GBC,'r-o',label='GradientBoostingClassifier')
# plt.plot(x,accuracy_GBR,'b-o',label='GradientBoostingRegressor')
# plt.plot(x,accuracy_RFC,'g-o',label='RandomForestClassifier')
# legend = plt.legend(loc='upper left', shadow=True, fontsize='x-small')
# legend.get_frame().set_facecolor('#00FFCC')
# plt.show()
# import matplotlib.pyplot as plt
#
# rate = [1, 7, 3, 9]
# explode = [0, 0, 0.1, 0]
# colors = ['c', 'm', 'y', 'g']
# labels = ['Apple', 'Pear', 'Peach', 'Orange']
#
# plt.pie(rate, explode=explode, colors=colors, labels=labels, autopct='%d%%')
# plt.show()

# temp=np.array([[1,2,3],[3,4,5]])
# print(np.square(temp))
# temp+=1
# print(temp)
#
# a=np.array([1,4,2,5,3])
# print(a.argsort())

# print(chr(65)+'2')

# a=np.array([[1,2,3],[2,3,4]])
# b=np.log(a)
# print(b)

# a=np.array([1,2,3])
# print(a+1)

