from com.cn.CFEEG import mergePCAFile2,groupByKByName,eigen,readFTM,mergeSlipper,mergeFile2,mergePCASlipper
from com.cn.Classifiers2 import *
import numpy as np
import os
import copy
import matplotlib.pyplot as plt

accuracy_GBC=[]
accuracy_GBR=[]
accuracy_RFC=[]
accuracy_SVM=[]
path='E:\EEG\MusicK'

kpaths=os.listdir(path)
kpaths=[os.path.join(path,p) for p in kpaths]


for kpath in kpaths:

    print("=========================================",kpath,"=========================================")
    files=os.listdir(kpath)
    filepaths=[os.path.join(kpath,filename) for filename in files]

    temp=None

    accuracy1=[]
    accuracy2=[]
    accuracy3=[]
    accuracy4=[]
    print('序号\t','迭代决策树\t','迭代回归\t','随即森林\t','SVM\t','验证集')
    for i in range(len(filepaths)):

        temp=copy.copy(filepaths)

        Source_test=temp[i]
        temp.remove(temp[i])

        test_filename=os.path.basename(Source_test)
        eigenvalue,eigenvector=eigen(temp)
        # print(eigenvector)

        testDataSet=mergePCASlipper([Source_test,],'第一组',row=(0,),col=(5,13),target_col=21,eigenvector=eigenvector)
        trainDataSet=mergePCASlipper(temp,'第一组',row=(0,),col=(5,13),target_col=21,eigenvector=eigenvector)

        X_train=trainDataSet['data']
        X_test=testDataSet['data']
        y_train=trainDataSet['target']
        y_test=testDataSet['target']

        result1=userGradientBoostingClassifier(X_train,X_test,y_train,y_test)
        result2=userGradientBoostingRegressor(X_train,X_test,y_train,y_test)
        result3=userRandomForestClassifier(X_train,X_test,y_train,y_test)
        result4=userSVMClassifier(X_train,X_test,y_train,y_test)

        accuracy1.append(result1)
        accuracy2.append(result2)
        accuracy3.append(result3)
        accuracy4.append(result4)
        print(str(i)+'\t',str(result1)+'\t',str(result2)+'\t',str(result3)+'\t',str(result4)+'\t'+test_filename)
    accuracy_GBC.append(np.mean(accuracy1))
    accuracy_GBR.append(np.mean(accuracy2))
    accuracy_RFC.append(np.mean(accuracy3))
    accuracy_SVM.append(np.mean(accuracy4))
    print('平均值'+'\t',str(np.mean(accuracy1))+'\t',str(np.mean(accuracy2))+'\t',str(np.mean(accuracy3))+'\t',str(np.mean(accuracy4)))

fig=plt.figure()
fig.suptitle('PCA Data Process')
axes=fig.add_subplot(111)

x=np.linspace(1.0,2.0,11)

axes.set_xlabel('K')
axes.set_ylabel('Accuracy Rate')
plt.plot(x,accuracy_GBC,'r-o',label='GradientBoostingClassifier')
plt.plot(x,accuracy_GBR,'b-o',label='GradientBoostingRegressor')
plt.plot(x,accuracy_RFC,'g-o',label='RandomForestClassifier')
plt.plot(x,accuracy_SVM,'m-o',label='SVMClassifier')
legend = plt.legend(loc='upper left', shadow=True, fontsize='x-small')
legend.get_frame().set_facecolor('#00FFCC')
plt.show()

















