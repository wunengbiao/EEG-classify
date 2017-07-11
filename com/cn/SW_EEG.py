from collections import defaultdict
import numpy as np
import os
def separate(a):
    split_index=[]
    for i in range(len(a)-1):
        if a[i]!=a[i+1]:
            split_index.append(i)
    return split_index

def groupByName(pre_dir,files):
    resultDict=defaultdict()
    for file in files:
        prefix=file.split('-')[0].lower()
        if prefix in resultDict:
            resultDict[prefix].append(os.path.join(pre_dir,file))
        else:
            resultDict[prefix]=[]
            resultDict[prefix].append(os.path.join(pre_dir,file))
    return resultDict

def split_third(arr):
    r,c=np.shape(arr)
    max=0
    index=0
    for i in range(1,r-1):
        m1=np.mean(arr[0:i])
        m2=np.mean(arr[i:])
        m=np.dot(m2-m1,m2-m1)
        s1=np.var(arr[0:i])
        s2=np.var(arr[i:])
        sw=np.sum(s1)+np.sum(s2)
        if m/sw>max:
            max=m/sw
            index=i
    print("该文件丢掉第三部分的数据的百分比为:",(r-index-1)/r)
    return index


testDataSet=np.array([[1,2,3,4],[2,3,4,5],[5,6,7,8],[3,1,2,4],[3,2,1,4]])

def normalization(dataSet):
    row,col=np.shape(dataSet)
    maxValue=np.max(dataSet,axis=0)
    minValue=np.min(dataSet,axis=0)
    divisor=maxValue-minValue

    diffMat=dataSet-np.tile(minValue,(row,1))
    return diffMat/divisor

# print(normalization(testDataSet))
def regularization(dataSet):
    row,col=np.shape(dataSet)

    norm2=np.sqrt(np.sum(dataSet*dataSet,axis=1))
    # print(norm2)
    divisor=np.tile(norm2,(col,1))
    # print(divisor)
    return dataSet/divisor.T

# print(regularization(testDataSet))
# regularization(testDataSet)








