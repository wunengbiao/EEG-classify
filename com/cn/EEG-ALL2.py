from com.cn.CFEEG import mergeFile2,regularizationMergeFile,normalizationMergeFile
from com.cn.Classifiers2 import userGradientBoostingRegressor,userGradientBoostingClassifier,userRandomForestClassifier,userSVMClassifier
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

accuracy_GBC=[]
accuracy_GBR=[]
accuracy_RFC=[]
accuracy_SVM=[]
accuracy_Log=[]
path='D:\EEG\理想数据k=1~2'

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
    accuracy5=[]
    print('序号\t','迭代决策树\t','迭代回归\t','随即森林\t','SVM\t','混合模型\t','验证集')
    for i in range(len(filepaths)):

        temp=copy.copy(filepaths)

        Source_test=temp[i]
        temp.remove(temp[i])
        test_filename = os.path.basename(Source_test)


        #ICA
        # testDataSet=mergeFile2([Source_test,],'ica_data',row=(0,),col=(0,8),target_col=8,window_size=10,step=3)
        # trainDataSet=mergeFile2(temp,'ica_data',row=(0,),col=(0,8),target_col=8,window_size=10,step=3)

        #普通
        # testDataSet=mergeFile2([Source_test,],'第一组',row=(0,),col=(5,13),target_col=21,window_size=10,step=3)
        # trainDataSet=mergeFile2(temp,'第一组',row=(0,),col=(5,13),target_col=21,window_size=10,step=3)

        #正则化
        # testDataSet = regularizationMergeFile([Source_test, ], '第一组', row=(0,), col=(5, 13), target_col=21, window_size=10, step=3)
        # trainDataSet = regularizationMergeFile(temp, '第一组', row=(0,), col=(5, 13), target_col=21, window_size=10, step=3)

        #归一化
        testDataSet = normalizationMergeFile([Source_test, ], '第一组', row=(0,), col=(5, 13), target_col=21, window_size=10, step=3)
        trainDataSet = normalizationMergeFile(temp, '第一组', row=(0,), col=(5, 13), target_col=21, window_size=10, step=3)

        # 原始数据
        # X_train=trainDataSet['data']
        # X_test=testDataSet['data']
        # y_train=trainDataSet['target']
        # y_test=testDataSet['target']

        # 指数形式
        # X_train = np.power(10,trainDataSet['data'])
        # X_test = np.power(10,testDataSet['data'])
        # y_train = trainDataSet['target']
        # y_test = testDataSet['target']

        #转化成对数对数形式

        X_train=np.log(trainDataSet['data']+0.01)
        X_test=np.log(testDataSet['data']+0.01)
        y_train=trainDataSet['target']
        y_test=testDataSet['target']

        train_half=int(X_train.shape[0]*3/4)
        # print(train_shape)


        clf1,result1,fGB,fGBT=userGradientBoostingClassifier(X_train[0:train_half,:],X_train[train_half:,:],y_train[0:train_half],y_train[train_half:],True)
        clf2,result2,fGBR,fGBRT=userGradientBoostingRegressor(X_train[0:train_half,:],X_train[train_half:,:],y_train[0:train_half],y_train[train_half:],True)
        clf3,result3,fRF,fRFT=userRandomForestClassifier(X_train[0:train_half,:],X_train[train_half:,:],y_train[0:train_half],y_train[train_half:],True)
        clf4,result4,fSVM,fSVMT=userSVMClassifier(X_train[0:train_half,:],X_train[train_half:,:],y_train[0:train_half],y_train[train_half:],True)

        feature=np.c_[fGBT,fGBRT,fRFT,fSVMT]
        test_data_log=np.c_[clf1.predict(X_test),clf2.predict(X_test),clf3.predict(X_test),clf4.predict(X_test)]

        # logistic_clf=LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='sag', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
        logistic_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2,
                                         max_depth=6, random_state=0).fit(X_train, y_train)
        logistic_clf.fit(feature,y_train[train_half:])
        result_log=logistic_clf.score(test_data_log,y_test)

        accuracy1.append(result1)
        accuracy2.append(result2)
        accuracy3.append(result3)
        accuracy4.append(result4)
        accuracy5.append(result_log)
        print(str(i)+'\t',str(result1)+'\t',str(result2)+'\t',str(result3)+'\t',str(result4)+'\t',str(result_log)+'\t'+test_filename)
    accuracy_GBC.append(np.mean(accuracy1))
    accuracy_GBR.append(np.mean(accuracy2))
    accuracy_RFC.append(np.mean(accuracy3))
    accuracy_SVM.append(np.mean(accuracy4))
    accuracy_Log.append(np.mean(accuracy5))
    print('平均值'+'\t',str(np.mean(accuracy1))+'\t',str(np.mean(accuracy2))+'\t',str(np.mean(accuracy3))+'\t',str(np.mean(accuracy4))+'\t',str(np.mean(accuracy5)))

fig=plt.figure()

fig.suptitle('Compare with Hybird model')
axes=fig.add_subplot(111)

x=np.linspace(1.0,2.0,11)

axes.set_xlabel('K')
axes.set_ylabel('Accuracy Rate')
axes.xaxis.grid(True, which='major')
axes.yaxis.grid(True, which='major')
# plt.plot(x,accuracy_GBC,'r-o',label='GradientBoostingClassifier')
# plt.plot(x,accuracy_GBR,'b-D',label='GradientBoostingRegressor')
# plt.plot(x,accuracy_RFC,'g-x',label='RandomForestClassifier')
# plt.plot(x,accuracy_SVM,'m->',label='SVMClassifier')
plt.plot(x,accuracy_Log,'c-s',label='HybirdModel')
legend = plt.legend(loc='upper left', shadow=True, fontsize='x-small')
legend.get_frame().set_facecolor('#00FFCC')
plt.show()
