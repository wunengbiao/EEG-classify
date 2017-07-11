import xlrd
import numpy as np
from collections import defaultdict
path="E:\脑电材料\数据采集\EEG-master\Analysis\EEG显著性分析_10s.xlsx"


# print(dataset)

# data=xlrd.open_workbook(path)
#
# sheet=data.sheet_by_index(1)
# print(sheet.nrows)
# dataset=np.zeros((sheet.nrows-1,8),dtype=float)
#
# # print(sheet.col(0))
# dataset[:,0]=sheet.col_values(0,1)
# print(dataset[0,0])
#
# # print(sheet.cell(1,0))
#
# # for i in range(8):
#
# a=defaultdict()
# print(a)
# a['target']=dataset
# print(a)

# print(sheet)

# print(np.array([1,2,3,4])/np.array([1,2,3,4]))
# a=(0,)
# print(a[0])
# print(a[1])

# b=[1,2,3,4,5,6,7,8]
# print(b[0:5])
#
# from com.cn.CFEEG import read_data
#
# result=read_data(path,'lrx6.20',row=(0,),col=(0,8))
# print(result)

# wave_dict={0:'Delta',1:'Theta',2:'Low Alpha',3:'High Alpha',4:'Low Beta',5:'High Beta',6:'Low Gamma',7:'Mid Gamma'}
#
# print(wave_dict[1])

# from com.cn.CFEEG import *
#
# result1=read_data(path,'lrx6.20',row=(0,),col=(0,8))
# print(result1)
# result2=data_processing(result1,[(0,2),(1,),(3,),(4,),(5,),(6,),(7,)])
# print(result2)
# result3=compute_corrcoef(result2)

# write_data(result3,'test','D:/test.xlsx')
# print(result3)
#
# import xlsxwriter
#
# book=xlsxwriter.Workbook('D:/test.xlsx')
# worksheet=book.add_worksheet('sheet1')

# data = ('Foo', 'Bar', 'Baz')

# Write the data to a sequence of cells.
# worksheet.write_row('A1', data)

# The above example is equivalent to:
# worksheet.write('A1', data[0])
# worksheet.write('B1', data[1])
# worksheet.write('C1', data[2])

# book.close()

import numpy as np

a=np.array([[1,2,3],[2,3,4]])
ret=np.transpose(np.var(a,axis=0))
# print(np.transpose(ret))
print(np.shape(ret))

print(np.r_[ret,ret])
