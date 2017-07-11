from com.cn.CFEEG import mergePCAFile

path="E:\EEG\music"
resultDict=mergePCAFile(path,'第一组',row=(0,),col=(13,21),target_col=21,window_size=10,step=3)

#-------------------------------------------------------------------------------------------------------
from sklearn import cross_validation
X_train,X_test,y_train,y_test=cross_validation.train_test_split(resultDict['data'],resultDict['target'],test_size=0.2,random_state=0)

#-------------------------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
# max_depth=15, min_samples_leaf=1,max_features=0.7
clf = RandomForestClassifier(n_estimators = 101,criterion='gini',min_samples_split=5,max_depth=10, min_samples_leaf=1,max_features=0.7,n_jobs=3 )

#训练模型
s = clf.fit(X_train , y_train)

#评估模型准确率
r = clf.score(X_test , y_test)
print('随机森林的准确率为:',r)
#-------------------------------------------------------------------------------------------------------
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2,
    max_depth=6, random_state=0).fit(X_train, y_train)
ret=clf.score(X_test, y_test)
print('迭代决策树:',ret)
print(clf.feature_importances_)

#-------------------------------------------------------------------------------------------------------
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