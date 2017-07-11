from com.cn.CFEEG import read_data,mergeSlipper

path="E:\脑电材料\数据采集\EEG-master\Data\全部数据\原始excel\平静\wsh早上2hz_20160602_103847_ASIC_EEG.xlsx"
path2="E:\脑电材料\数据采集\EEG-master\Data\全部数据\原始excel\平静\lz-调节2hz_20160607_102336_ASIC_EEG.xlsx"
path3="E:\脑电材料\数据采集\EEG-master\Data\全部数据\原始excel\平静\lz-调节2hz_20160608_111244_ASIC_EEG.xlsx"
path4="E:\脑电材料\数据采集\EEG-master\Data\全部数据\原始excel\平静\wnb-201608281115_20160829_111545_ASIC_EEG.xlsx"
dataset1=read_data(path,'第一组',row=(0,),col=(13,21),target_col=21)
dataset2=read_data(path2,'第一组',row=(0,),col=(13,21),target_col=21)
dataset3=read_data(path3,'第一组',row=(0,),col=(13,21),target_col=21)
dataset4=read_data(path4,'第一组',row=(0,),col=(13,21),target_col=21)

# print(dataset2['header'])

data1=mergeSlipper(dataset1,10,3)
print(data1['header'])
new_data1=data1['data']

data2=mergeSlipper(dataset2,10,3)
new_data2=data2['data']

data3=mergeSlipper(dataset3,10,3)
new_data3=data3['data']

data4=mergeSlipper(dataset4,10,3)
new_data4=data4['data']

target1=data1['target']
target2=data2['target']
target3=data3['target']
target4=data4['target']

import numpy as np
new_data=np.r_[new_data1,new_data2,new_data3,new_data4]
target=target1+target2+target3+target4

from sklearn import cross_validation

X_train,X_test,y_train,y_test=cross_validation.train_test_split(new_data,target,test_size=0.2,random_state=0)


from sklearn import svm

clf=svm.SVC(kernel='rbf')
clf.fit(X_train,y_train)

result=clf.predict(X_test)
count=0
for i in range(len(y_test)):
    if result[i]==y_test[i]:
        count+=1
print('SVM的准确率为:',count/len(y_test))


from sklearn.cluster import KMeans
from sklearn import metrics
kmeans_model=KMeans(n_clusters=2,random_state=1).fit(X_train)
count=0
result=kmeans_model.predict(X_test)
for i in range(len(y_test)):
    if result[i]==y_test[i]:
        count+=1
print('K-Means的准确率为:',max(count/len(y_test),1-count/len(y_test)))

from sklearn.ensemble import RandomForestClassifier
# max_depth=15, min_samples_leaf=1,max_features=0.7
clf = RandomForestClassifier(n_estimators = 101,criterion='entropy',min_samples_split=5 )

#训练模型
s = clf.fit(X_train , y_train)

#评估模型准确率
r = clf.score(X_test , y_test)
print('随机森林的准确率为:',r)


from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=7, min_samples_split=20, min_samples_leaf=5,
min_weight_fraction_leaf=0.0, max_features=0.7, random_state=None,max_leaf_nodes=None,class_weight=None, presort=False)

clf=clf.fit(X_train,y_train)

result=clf.predict(X_test)
count=0
for i in range(len(y_test)):
    if result[i]==y_test[i]:
        count+=1

print('决策树的准确率为:',count/len(y_test))

# from sklearn.externals.six import StringIO
# import pydot
# from IPython.display import Image
# dot_data=StringIO()
# from sklearn import tree
#
# target_name=['Before','After']
#
# tree.export_graphviz(clf, out_file=dot_data,
#                          feature_names=data1['header'],
#                          class_names=target_name,
#                          filled=True, rounded=True,
#                          special_characters=True)
# graph=pydot.graph_from_dot_data(dot_data.getvalue())
# graph[0].write_pdf("eeg.pdf")
