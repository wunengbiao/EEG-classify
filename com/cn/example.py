import numpy as np

a=np.array([[1,2,3,4],[4,5,6,7],[7,8,9,10]])
b=np.array([5,6,7,8])

# print(np.shape(a.T))
# print(np.shape(a))
#
# m1=np.mean(a,axis=0)
# print(m1.T)
# print(np.shape(m1))
# print(np.dot(m1.T,m1))

a=np.array([[1,2,3,4]])

r=np.shape(a)
print(r)
b=np.array([[1],[2],[3],[4]])

c=np.dot(a,b)
print(np.shape(c))

x=np.array([1,2,3,4])
print(np.dot(x,x))

y=np.sum(x)
print(y)

from com.cn.SW_EEG import split_third

arr=np.array([[1,2,3,4],[1,3,2,5],[3,4,5,6],[5,4,3,2],[100,100,100,100]])
index=split_third(arr)
print(index)