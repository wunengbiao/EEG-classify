from com.cn.CFEEG import mergeFile

path="E:\脑电材料\数据采集\EEG-master\Data\全部数据\WNBG"
resultDict=mergeFile(path,'第一组',row=(0,),col=(13,21),target_col=21,window_size=10,step=3)

pathTest="E:\脑电材料\数据采集\EEG-master\Data\全部数据\WNB"
resultDictTest=mergeFile(pathTest,'第一组',row=(0,),col=(13,21),target_col=21,window_size=10,step=3)
# print(resultDict)
import numpy as np
print(np.shape(resultDict['data']))

X_train=resultDict['data']
y_train=resultDict['target']
X_test=resultDictTest['data']
y_test=resultDictTest['target']
from sklearn.ensemble import RandomForestClassifier
# max_depth=15, min_samples_leaf=1,max_features=0.7
clf = RandomForestClassifier(n_estimators = 101,criterion='gini',min_samples_split=5,max_depth=10, min_samples_leaf=1,max_features=0.7,n_jobs=3 )

#训练模型
s = clf.fit(X_train , y_train)

#评估模型准确率
r = clf.score(X_test , y_test)
print('随机森林的准确率为:',r)

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2,
    max_depth=6, random_state=0).fit(X_train, y_train)
ret=clf.score(X_test, y_test)
print('迭代决策树:',ret)
print(clf.feature_importances_)

from sklearn.ensemble import GradientBoostingRegressor

clf=GradientBoostingRegressor(n_estimators=100,max_depth=6,random_state=0)
clf.fit(X_train,y_train)
ret=clf.predict(X_test)
count=0

l=len(y_test)

count=0
for i in range(l):
    if (ret[i]<0.5 and y_test[i]==0) or (ret[i]>=0.5 and y_test[i]==1):
        count+=1
print("迭代回归:",count/l)