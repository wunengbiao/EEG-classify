from com.cn.CFEEG import read_data,data_processing
import numpy as np
from collections import defaultdict
'''
    重心频率  Delta/low alpha Delta/High alpha  Delta/low beta   Delta/High beta
'''

path="E:\脑电材料\数据采集\EEG-master\Data\全部数据\原始excel\平静\cxq单点2hz_20160623_102453_ASIC_EEG.xlsx"

weight=[2,5.5,8.5,11,15,24,35.5,45.5]
wave_dict={0:'Delta',1:'Theta',2:'Low Alpha',3:'High Alpha',4:'Low Beta',5:'High Beta',6:'Low Gamma',7:'Mid Gamma'}

result=read_data(path,'第一组',row=(0,),col=(5,13),target_col=21)
dataSet1=result['data']

gravity=np.dot(dataSet1,np.array(weight))
#

result2=data_processing(result,[(0,2),(0,3),(0,4),(0,5)])
dataSet2=result2['data']
# print(np.shape(dataSet2))

# print(dataSet2['target'])

new_data=np.c_[dataSet2,gravity]
# feature_names=dataSet2['target'].append('Gravity Frequency')
feature_names=result2['header']
feature_names.append('Gravity Frequency')

# print(feature_names)
# print(new_data)
# print(np.shape(new_data))
target=result2['target']
# print(np.shape(target))
# print(target)
'''
    处理new_data,feature_names,target
'''

from sklearn import cross_validation

X_train,X_test,y_train,y_test=cross_validation.train_test_split(dataSet2,target,test_size=0.3,random_state=0)

from sklearn.cluster import KMeans
from sklearn import metrics
kmeans_model=KMeans(n_clusters=2,random_state=1).fit(X_train)

# labels=kmeans_model.labels_
# ret=metrics.silhouette_score(X_train,labels,metric='euclidean')
# print('k-means的准确率:',ret)
count=0

result=kmeans_model.predict(X_test)
# print(result)
for i in range(len(y_test)):
    if result[i]==y_test[i]:
        count+=1
print('K-Means的准确率为:',count/len(y_test))


from sklearn import svm

clf=svm.SVC(degree=4)
clf.fit(X_train,y_train)

result=clf.predict(X_test)
count=0
for i in range(len(y_test)):
    if result[i]==y_test[i]:
        count+=1
print('SVM的准确率为:',count/len(y_test))



from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier(criterion='gini',min_samples_split=5)

clf=clf.fit(X_train,y_train)

result=clf.predict(X_test)
count=0
for i in range(len(y_test)):
    if result[i]==y_test[i]:
        count+=1

print('决策树的准确率为:',count/len(y_test))

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)

result=classifier.predict(X_test)

print('Logistic Regression准确率为:',classifier.score(X_test,y_test))