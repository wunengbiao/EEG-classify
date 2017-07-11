from com.cn.CFEEG import read_data,split_data,mergeData,data_processing
path="E:\脑电材料\数据采集\EEG-master\Data\全部数据\原始excel\平静\cxq单点2hz_20160623_102453_ASIC_EEG.xlsx"
path2="E:\脑电材料\数据采集\EEG-master\Data\全部数据\原始excel\平静\wsh早上2hz_20160602_103847_ASIC_EEG.xlsx"
path3="E:\脑电材料\数据采集\EEG-master\Data\全部数据\原始excel\平静\lz-调节2hz_20160608_111244_ASIC_EEG.xlsx"

# dataset=read_data(path,'第一组',row=(0,),col=(5,13),target_col=21)
# print(dataset)
# result=split_data(dataset)
# print(split_data(dataset))
import numpy as np
# print(result['data'])
# print(np.shape(result['target']))
# print(result['target'])

result=mergeData([path,path2,path3],['第一组','第一组','第一组'],[(0,),(0,),(0,)],[(5,13),(5,13),(5,13)],[21,21,21])
# print(result)
# print(np.shape(result['data']))

weight=[2,5.5,8.5,11,15,24,35.5,45.5]
wave_dict={0:'Delta',1:'Theta',2:'Low Alpha',3:'High Alpha',4:'Low Beta',5:'High Beta',6:'Low Gamma',7:'Mid Gamma'}

dataSet1=result['data']

gravity=np.dot(dataSet1,np.array(weight))

result2=data_processing(result,[(0,),(1,),(2,),(3,),(4,),(5,),(6,),(7,),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(2,3),(2,4),(2,5)
    , (2, 5),(2,7),(3,4),(3,5),(3,6),(3,7),(4,5),(4,6),(4,7),(5,6),(5,7),(6,7)])
dataSet2=result2['data']
for i in range(10):
    print(dataSet2[i,:])

new_data=np.c_[dataSet2,gravity]

print(type(result2['header']))
feature_names=result2['header']
feature_names.append('Gravity Frequency')
target=result2['target']

from sklearn import cross_validation

X_train,X_test,y_train,y_test=cross_validation.train_test_split(new_data,target,test_size=0.3,random_state=0)

from sklearn.cluster import KMeans
from sklearn import metrics
kmeans_model=KMeans(n_clusters=2,random_state=1).fit(X_train)
count=0
result=kmeans_model.predict(X_test)
for i in range(len(y_test)):
    if result[i]!=y_test[i]:
        count+=1
print('K-Means的准确率为:',count/len(y_test))


from sklearn import svm

clf=svm.SVC(kernel='rbf')
clf.fit(X_train,y_train)

result=clf.predict(X_test)
count=0
for i in range(len(y_test)):
    if result[i]==y_test[i]:
        count+=1
print('SVM的准确率为:',count/len(y_test))

from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier(criterion='gini',min_samples_split=50)

clf=clf.fit(X_train,y_train)

result=clf.predict(X_test)
count=0
for i in range(len(y_test)):
    if result[i]==y_test[i]:
        count+=1

print('决策树的准确率为:',count/len(y_test))
