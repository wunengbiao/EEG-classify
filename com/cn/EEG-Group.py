from com.cn.CFEEG import mergePCAFile2,groupByKByName,eigenV,mergeFile2
from com.cn.Classifiers2 import userGradientBoostingClassifier,userGradientBoostingRegressor,userRandomForestClassifier
import numpy as np
import os

path='E:\\EEG\\k'

KGroup=groupByKByName(path)

for kValue in KGroup.keys():
    print('================================',kValue,'==========================')
    eigenvalue,eigenvector=eigenV(os.path.join(path,kValue))
    for nameKey in KGroup[kValue].keys():
        print('                     ',nameKey)
        print('----------------------------------------------------------------')
        files=KGroup[kValue][nameKey]

        accuracy1 = []
        accuracy2 = []
        accuracy3 = []
        print('\t', '迭代回归\t\t\t\t', '随机森林\t\t\t\t', '迭代决策树')
        for i in range(len(files)):


            testDict = mergePCAFile2([files[i],], '第一组', row=(0,), col=(13, 21), target_col=21, window_size=10, step=3,eigenvalue=eigenvalue,eigenvector=eigenvector)
            paths=[]
            for j in range(len(files)):
                if j!=i:
                    paths.append(files[j])
            trainDict=mergePCAFile2(paths,'第一组', row=(0,), col=(13, 21), target_col=21, window_size=10, step=3,eigenvalue=eigenvalue,eigenvector=eigenvector)
            X_train=trainDict['data']
            y_train=trainDict['target']
            X_test=testDict['data']
            y_test=testDict['target']

            score1=userGradientBoostingRegressor(X_train,X_test,y_train,y_test)
            score2=userRandomForestClassifier(X_train,X_test,y_train,y_test)
            score3=userGradientBoostingClassifier(X_train,X_test,y_train,y_test)
            accuracy1.append(score1)
            accuracy2.append(score2)
            accuracy3.append(score3)
            print(i,'\t',score1,'\t',score2,'\t',score3)
        print('平均值:\t',np.mean(accuracy1),'\t',np.mean(accuracy2),'\t',np.mean(accuracy3))
        print('--------------------------------------------------------------')




