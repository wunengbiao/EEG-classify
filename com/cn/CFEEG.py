import xlsxwriter
import xlrd
import sys
import os
from com.cn.SW_EEG import groupByName,regularization,normalization

import numpy as np

from collections import defaultdict
'''
    [Delta,Theta,Low Alpha,High Alpha,Low Beta,High Beta,Low Gamma,Mid Gamma]

'''

wave_dict={0:'Delta',1:'Theta',2:'Low Alpha',3:'High Alpha',4:'Low Beta',5:'High Beta',6:'Low Gamma',7:'Mid Gamma'}

'''
    row=(start,end)
    col=(start,end)
'''
def read_data(path,sheet,*,row,col,target_col):
    result=defaultdict()

    data=xlrd.open_workbook(path)
    work_sheet=data.sheet_by_name(sheet)

    row_start=row[0]
    if len(row)==1:
        row_end=work_sheet.nrows-1
    else:
        row_end=row[1]

    col_start=col[0]
    if len(col)==1:
        col_end=work_sheet.ncols-1
    else:
        col_end=col[1]


    header=work_sheet.row_values(row_start)[col_start:col_end]
    result['header']=header

    size_row=row_end-row_start
    size_col=col_end-col_start

    dataSet=np.zeros((size_row,size_col))

    for i in range(1,size_row+1):
        dataSet[i-1,:]=work_sheet.row_values(i)[col_start:col_end]

    result['data']=dataSet

    target=work_sheet.col_values(target_col,row_start+1,row_end+1)
    result['target']=target

    return result

'''
    resultDict:defaultdict
    condition:List<Tuple>
'''

def data_processing(resultDict,condition):

    result=defaultdict()
    size=len(condition)

    header=[]
    for i in range(size):
        if len(condition[i])==1:
            header.append(wave_dict[condition[i][0]])
        else:
            tmp=wave_dict[condition[i][0]]+'/'+wave_dict[condition[i][1]]
            header.append(tmp)
    result['header']=header

    target=resultDict['target']
    result['target']=target

    data = resultDict['data']
    shape = np.shape(data)

    dataSet=np.zeros((shape[0],size),dtype=float)

    for i in range(size):
        if len(condition[i])==1:
            dataSet[:,i]=data[:,condition[i][0]]
        else:
            dataSet[:,i]=data[:,condition[i][0]]/data[:,condition[i][1]]
    result['data']=dataSet

    return result

'''
    抽取取调节前后的数据
'''

from com.cn.SW_EEG import separate,split_third
def split_data(resultDcit):
    result=defaultdict()
    target=resultDcit['target']
    [F,S]=separate(target)
    # print(F,'----indexF')
    dataSet=resultDcit['data']
    data=np.r_[dataSet[0:F+1],dataSet[S+1:]]
    F_target=target[0:F+1]
    S_target=[(i+1) for i in target[S+1:]]
    new_target=F_target+S_target
    result['header']=resultDcit['header']
    result['data']=data
    result['target']=new_target
    return result

'''
滑动窗口滑动数据
'''

def mergeSlipper(resultDict,windows_size,step):
    result=defaultdict()

    t=resultDict['target']
    [F,S]=separate(t)

    dataSet=resultDict['data']
    Fdata=dataSet[0:F+1]
    Sdata=dataSet[S+1:]

    FAdata=slipper(Fdata,windows_size,step)
    SAdata=slipper(Sdata,windows_size,step)

    new_data=np.r_[FAdata,SAdata]

    Fr, Fc = np.shape(FAdata)
    Sr, Sc = np.shape(SAdata)
    target=[0 for i in range(Fr)]+[1 for i in range(Sr)]

    h=resultDict['header']
    header=[]
    for i in h:
        th='std of '+i
        header.append(th)
    for i in h:
        th='median of '+i
        header.append(th)
    for i in h:
        th='mean of '+i
        header.append(th)
    for i in h:
        th='min of '+i
        header.append(th)
    for i in h:
        th='max of '+i
        header.append(th)

    result['header']=header
    result['data']=new_data
    result['target']=target
    return result

def slipper(dataSet,windows_size,step):

    r,c=np.shape(dataSet)

    index=0
    temp1=np.std(dataSet[index:windows_size,:],axis=0)
    temp2=np.median(dataSet[index:windows_size,:],axis=0)
    temp3=np.mean(dataSet[index:windows_size,:],axis=0)
    temp4=np.min(dataSet[index:windows_size,:],axis=0)
    temp5=np.max(dataSet[index:windows_size,:],axis=0)
    """
        变化系数
    """
    temp6=temp1/temp3

    temp=np.r_[temp1,temp2,temp3,temp4,temp5,temp6]
    # print(np.shape(temp.T))
    index+=step

    while index+windows_size<r:
        tempC=np.r_[np.std(dataSet[index:index+windows_size,:],axis=0),np.median(dataSet[index:index+windows_size,:],axis=0),np.mean(dataSet[index:index+windows_size,:],axis=0),
                    np.min(dataSet[index:index + windows_size, :], axis=0),np.max(dataSet[index:index+windows_size,:],axis=0),np.std(dataSet[index:index+windows_size,:],axis=0)/np.mean(dataSet[index:index+windows_size,:],axis=0)]
        temp=np.c_[temp,tempC]
        index+=step

    return temp.T

'''
    合并多个文件组成一个大的数据集
'''
def mergeData(paths,sheets,rows,cols,target_cols):

    resultCollect=[]

    resultDict=defaultdict()


    for i in range(len(paths)):
        result=read_data(paths[i],sheets[i],row=rows[i],col=cols[i],target_col=target_cols[i])
        resultCollect.append(split_data(result))

    header=resultCollect[i]['header']

    temp=resultCollect[0]['data']
    target=[]
    for i in range(1,len(resultCollect)):
        temp=np.r_[temp,resultCollect[i]['data']]

    for result in resultCollect:
        target=target+result['target']

    resultDict['header']=header
    resultDict['target']=target
    resultDict['data']=temp
    return resultDict



def compute_corrcoef(resultDict):

    result=defaultdict()
    data=resultDict['data']
    dataSet=np.corrcoef(data.T)

    result['corrcoef']=dataSet

    header=resultDict['header']
    result['header']=header

    return result


def write_data(resultDict,sheet,path):

    corrcoef_matrix=resultDict['corrcoef']
    header=resultDict['header']

    workbook=xlsxwriter.Workbook(path)
    work_sheet=workbook.add_worksheet(sheet)
    work_sheet.write_row('A1',header)

    shape=np.shape(corrcoef_matrix)

    for i in range(shape[0]):
        position='A'+str(i+2)
        work_sheet.write_row(position,corrcoef_matrix[i,:])

    workbook.close()


import os
def mergeFile(path,sheet,*,row,col,target_col,window_size,step):

    paths=os.listdir(path)
    result=defaultdict()
    result['target']=[]
    # result['data']=None

    for filename in paths:
        # print(os.path.join(path,filename),'开始处理！！！')

        dataset1=read_data(os.path.join(path,filename),sheet,row=row,col=col,target_col=target_col)
        new_data=mergeSlipper(dataset1,window_size,step)
        result['header']=new_data['header']
        result['target']=result['target']+new_data['target']
        # if result['data']==None:
        #     result['data']=new_data['data']
        # else:
        #     result['data']=np.r_[result['data'],new_data['data']]

        if 'data' in result.keys():
            result['data'] = np.r_[result['data'], new_data['data']]
        else:
            result['data'] = new_data['data']

        # print(os.path.join(path,filename),'处理成功！！！')

    return result

def pca(n_feature,path,sheet,*,row,col,target_col,eigenvalue,eigenvector):
    resultDict=defaultdict()

    header=[]
    for i in range(n_feature):
        header.append('FP'+str(i+1))
    resultDict['header']=header

    dataSet=read_data(path,sheet,row=row,col=col,target_col=target_col)

    target=dataSet['target']
    resultDict['target']=target

    data=dataSet['data']

    # row,col=np.shape(data)
    # u=np.mean(data,axis=0)
    # diffMat=data-np.tile(u,(row,1))
    # CovX=np.dot(diffMat.T,diffMat)

    # print("特征占比:",np.sum(eigenvalue[0:n_feature]/np.sum(eigenvalue)))
    new_data=np.dot(data,eigenvector[0:n_feature,:].T)
    resultDict['data']=new_data
    return resultDict

def eigenV(path):
    pcaDict = mergeFile(path, '第一组', row=(0,), col=(13, 21), target_col=21, window_size=10, step=3)
    data=pcaDict['data'][:,0:8]
    row, col = np.shape(data)
    u = np.mean(data, axis=0)
    diffMat = data - np.tile(u, (row, 1))
    CovX = np.dot(diffMat.T, diffMat)/(row-1)
    eigenvalue, eigenvector = np.linalg.eig(CovX)
    return eigenvalue,eigenvector

def mergePCAFile(path,sheet,*,row,col,target_col,window_size,step,eigenvalue,eigenvector):

    paths=os.listdir(path)
    result=defaultdict()
    result['target']=[]
    result['data']=None

    for filename in paths:
        print(os.path.join(path,filename),'开始处理！！！')

        dataset1=pca(4,os.path.join(path,filename),sheet,row=row,col=col,target_col=target_col,eigenvalue=eigenvalue,eigenvector=eigenvector)
        new_data=mergeSlipper(dataset1,window_size,step)
        result['header']=new_data['header']
        result['target']=result['target']+new_data['target']
        if result['data']==None:
            result['data']=new_data['data']
        else:
            result['data']=np.r_[result['data'],new_data['data']]

        print(os.path.join(path,filename),'处理成功！！！')
    return result

def mergePCAFile2(paths,sheet,*,row,col,target_col,window_size,step,eigenvalue,eigenvector):

    result=defaultdict()
    result['target']=[]
    # result['data']=None

    for path in paths:
        # print(path,'------ begin to process ------')

        dataset1=pca(4,path,sheet,row=row,col=col,target_col=target_col,eigenvalue=eigenvalue,eigenvector=eigenvector)
        new_data=mergeSlipper(dataset1,window_size,step)
        result['header']=new_data['header']
        result['target']=result['target']+new_data['target']
        # if result['data']==None:
        #     result['data']=new_data['data']
        # else:
        #     result['data']=np.r_[result['data'],new_data['data']]

        if 'data' in result.keys():
            result['data'] = np.r_[result['data'], new_data['data']]
        else:
            result['data'] = new_data['data']

        # print(path,'------- process successfully ------')
    return result

def groupByKByName(path):
    dirs=os.listdir(path)
    pathGroupByKValue=[]
    resultDict=defaultdict()
    for i in range(len(dirs)):
        pathGroupByKValue.append(os.path.join(path,dirs[i]))
        for kDir in pathGroupByKValue:
            files=os.listdir(kDir)
            namesDict=groupByName(kDir,files)
            resultDict[dirs[i]]=namesDict
    return resultDict


def readFTM(paths,sheet,*,row,col,target_col):
    resultDict=defaultdict()
    resultDict['target']=[]

    for path in paths:
        dataSet=read_data(path,sheet,row=row,col=col,target_col=target_col)
        target=dataSet['target']
        [F,S]=separate(target)
        F_target=target[0:F+1]
        S_target=[(i+1) for i in target[S+1:]]
        new_target=F_target+S_target
        resultDict['target']=resultDict['target']+new_target

        data=dataSet['data']
        F_data=data[0:F+1,:]
        S_data=data[S+1:,:]
        new_data=np.r_[F_data,S_data]

        if 'data' in resultDict.keys():
            resultDict['data']=np.r_[resultDict['data'],new_data]
        else:
            resultDict['data']=new_data

        # if 'source_data' in resultDict.keys():
        #     resultDict['source_data']=np.r_[resultDict['source_data'],data]
        # else:
        #     resultDict['source_data']=data

    return resultDict

def mergePCASlipper(paths,sheet,*,row,col,target_col,eigenvector):
    resultDict = defaultdict()
    resultDict['target'] = []

    for path in paths:
        dataSet=read_data(path,sheet,row=row,col=col,target_col=target_col)
        target=dataSet['target']
        [F,S]=separate(target)

        data=dataSet['data']
        F_data=data[0:F+1,:]
        S_data=data[S+1:,:]

        # print(np.shape(F_data),np.shape(S_data),'------------->')

        F_PCA_DATA=np.dot(F_data,eigenvector[0:4].T)
        S_PCA_DATA=np.dot(S_data,eigenvector[0:4].T)

        F_S_PCA_DATA=slipper(F_PCA_DATA,10,3)
        S_S_PCA_DATA=slipper(S_PCA_DATA,10,3)

        Fr,Fc=np.shape(F_S_PCA_DATA)
        Sr,Sc=np.shape(S_S_PCA_DATA)

        pca_target=[0 for i in range(Fr)]+[1 for j in range(Sr)]

        resultDict['target']=resultDict['target']+pca_target



        if 'data' in resultDict.keys():
            resultDict['data']=np.r_[resultDict['data'],F_S_PCA_DATA,S_S_PCA_DATA]
        else:
            resultDict['data']=np.r_[F_S_PCA_DATA,S_S_PCA_DATA]

    return resultDict



def eigen(paths):
    pcaDict = readFTM(paths,'第一组',row=(0,),col=(5,13),target_col=21)
    data=pcaDict['data']
    row, col = np.shape(data)
    u = np.mean(data, axis=0)
    diffMat = data - np.tile(u, (row, 1))
    CovX = np.dot(diffMat.T, diffMat)/(row-1)
    eigenvalue, eigenvector = np.linalg.eig(CovX)
    return eigenvalue,eigenvector

def mergeFile2(paths,sheet,*,row,col,target_col,window_size,step):

    result=defaultdict()
    result['target']=[]
    # result['data']=None

    for path in paths:
        # print(os.path.join(path,filename),'开始处理！！！')

        dataset1=read_data(path,sheet,row=row,col=col,target_col=target_col)
        new_data=mergeSlipper(dataset1,window_size,step)
        result['header']=new_data['header']
        result['target']=result['target']+new_data['target']
        # if result['data']==None:
        #     result['data']=new_data['data']
        # else:
        #     result['data']=np.r_[result['data'],new_data['data']]

        if 'data' in result.keys():
            result['data'] = np.r_[result['data'], new_data['data']]
        else:
            result['data'] = new_data['data']

        # print(os.path.join(path,filename),'处理成功！！！')

    return result

def regularizationMergeFile(paths,sheet,*,row,col,target_col,window_size,step):

    result=defaultdict()
    result['target']=[]

    for path in paths:

        dataset1=read_data(path,sheet,row=row,col=col,target_col=target_col)
        data=regularization(dataset1['data'])
        dataset1['data']=data
        new_data=mergeSlipper(dataset1,window_size,step)
        result['header']=new_data['header']
        result['target']=result['target']+new_data['target']

        if 'data' in result.keys():
            result['data'] = np.r_[result['data'], new_data['data']]
        else:
            result['data'] = new_data['data']

    return result

def normalizationMergeFile(paths,sheet,*,row,col,target_col,window_size,step):

    result=defaultdict()
    result['target']=[]

    for path in paths:

        dataset1=read_data(path,sheet,row=row,col=col,target_col=target_col)
        data=normalization(dataset1['data'])
        dataset1['data']=data
        new_data=mergeSlipper(dataset1,window_size,step)
        result['header']=new_data['header']
        result['target']=result['target']+new_data['target']

        if 'data' in result.keys():
            result['data'] = np.r_[result['data'], new_data['data']]
        else:
            result['data'] = new_data['data']

    return result

from sklearn.decomposition.fastica_ import FastICA

def ica(data):
    fast_ica=FastICA(8,fun='exp',max_iter=1000)
    new_data=fast_ica.fit_transform(data)
    return new_data


























