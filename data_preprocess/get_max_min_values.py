# 获取各种数据的最大值和最小值并存储，用于数据标准化
from data_preprocess.database_connect import *
import datetime
import numpy as np
import pandas as pd
import os


# 获取文件1和文件2的最大值和最小值
def getMaxMinValues(file1,file2=''):
    max_val = 0
    min_val = 100
    csvFile = pd.read_csv(file1)
    # print(csvFile.loc[1][0])
    for i in range(csvFile.shape[0]):
        for j in range(1,csvFile.shape[1]):
            if csvFile.loc[i][j] > max_val:
                max_val = csvFile.loc[i][j]
            if csvFile.loc[i][j] < min_val:
                min_val = csvFile.loc[i][j]
    if file2 == '':
        return max_val,min_val

    csvFile = pd.read_csv(file2)
    for i in range(csvFile.shape[0]):
        for j in range(1,csvFile.shape[1]):
            if csvFile.loc[i][j] > max_val:
                max_val = csvFile.loc[i][j]
            if csvFile.loc[i][j] < min_val:
                min_val = csvFile.loc[i][j]
    return max_val, min_val
