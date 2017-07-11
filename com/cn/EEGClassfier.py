from com.cn.CFEEG import *
import numpy as np
from sklearn import tree
path="E:\EEG\理想数据k=1~2\k=1.0\hzj-调节2hz_20160903_113852_ASIC_EEG.xlsx"
dataset=read_data(path,'第一组',row=(0,),col=(5,13),target_col=21)
# print(dataset)

feature_name=dataset['header']
data=dataset['data']
print()
target=dataset['target']

# print(feature_name)
# print(data)
# print(target)
# print(np.shape(target))
# print(np.shape(data))

from sklearn import cross_validation

X_train,X_test,y_train,y_test=cross_validation.train_test_split(data,target,test_size=0.3,random_state=0)

from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier(max_depth=4,criterion='gini',min_samples_split=3)

clf=clf.fit(X_train,y_train)

result=clf.predict(X_test)
# print(result)
# print(result.score)
from IPython.display import Image
# from sklearn.externals.six import StringIO
# import pydot
#
# dot_data = StringIO()
#
# tree.export_graphviz(clf, out_file=dot_data,
#                          feature_names=feature_name,
#                          filled=True, rounded=True,
#                          special_characters=True)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("iris.pdf")

count=0
for i in range(len(y_test)):
    if result[i]==y_test[i]:
        count+=1

print('决策树的准确率为:',count/len(y_test))

from sklearn.cluster import KMeans
from sklearn import metrics
kmeans_model=KMeans(n_clusters=3,random_state=1).fit(X_train)

# labels=kmeans_model.predict(X_test)
# labels=kmeans_model.labels_
# ret=metrics.silhouette_score(X_test,labels,metric='euclidean')
# print('k-means的准确率:',ret)
# print(labels)

count=0

result=kmeans_model.predict(X_test)
print(result)
for i in range(len(y_test)):
    if result[i]==y_test[i]:
        count+=1
print('K-Means的准确率为:',count/len(y_test))

from sklearn import svm
from com.cn.Classifiers2 import userSVMClassifier
score=userSVMClassifier(X_train,X_test,y_train,y_test)
print('SVM的准确率为:',score)




