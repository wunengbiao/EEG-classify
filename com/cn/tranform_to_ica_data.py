import numpy as np
import os
from com.cn.CFEEG import read_data
from sklearn.decomposition.fastica_ import FastICA
import xlsxwriter


def write_xlsx(header,data, target, path):
    workbook = xlsxwriter.Workbook(path)
    header.append('是否调节')
    work_sheet = workbook.add_worksheet('ica_data')
    work_sheet.write_row('A1', header)


    for i in range(8):

        position = chr(65+i)
        position=position+str(2)
        work_sheet.write_column(position,data[:,i])

    work_sheet.write_column('I2',target)

    workbook.close()



def transform(path_source, path_destination):
    files = os.listdir(path_source)

    for file in files:
        resultDict = read_data(os.path.join(path_source, file), '第一组', row=(0,), col=(5, 13), target_col=21)
        data = resultDict['data']
        target = resultDict['target']
        header=resultDict['header']

        fast_ica = FastICA(8, fun='exp', max_iter=1000)
        new_data = fast_ica.fit_transform(data)

        temp = new_data
        ava_data = np.mean(data, axis=0)
        order = []
        for i in range(8):
            H = np.c_[temp, temp * (-1)]
            H = H + 2
            H *= (ava_data[i] / 2)
            min = float('inf')
            index = -1
            for j in range(16):
                if j in order or (j - 8) in order: continue
                a = np.linalg.norm(data[:, i] - H[:, j], 1)
                if a < min:
                    min = a
                    index = j
            if index >= 8:
                index = index - 8
            order.append(index)

        ica_order = np.array(order).argsort()
        ica_data = np.c_[
            new_data[:, ica_order[0]], new_data[:, ica_order[1]], new_data[:, ica_order[2]], new_data[:, ica_order[
                3]], new_data[:, ica_order[4]], new_data[:, ica_order[5]], new_data[:, ica_order[6]], new_data[:,
                                                                                                      ica_order[7]]]
        write_xlsx(header,ica_data, target, os.path.join(path_destination, file))

transform('E:\EEG\理想数据k=1~2\k=1.9',"E:\EEG\\new\k=1.9")