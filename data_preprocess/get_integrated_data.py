# 该文件用于获取模型训练和验证的整合的数据（标准化处理），并存储到文件中
from data_preprocess.database_connect import *
from data_preprocess.get_max_min_values import getMaxMinValues
import datetime
import numpy as np
import pandas as pd
import os


# 获取用于训练的整合的数据并存储
# id:仓库序号（1~30）
# user_type: churner或loyaler
# period_length: 120或30
# overlap_ratio: 当user_type为loyaler时，获取数据时同一开发者不同区间的重叠度
# 注：period_length和overlap_ratio共同用于确定文件版本号
# detailed_data_dir: 存储训练样本详细数据的文件夹
# data_type_list:整合的数据类型列表
def getIntegratedDataAndSave(detailed_data_dir, save_dir, user_type, period_length, overlap_ratio,
                             data_type_list=None):
    if data_type_list is None:
        data_type_list = [
            'issue',
            'issue comment',
            'pull',
            'pull merged',
            # 'review',########
            'review comment',
            # 'commit',
            'betweeness',
            'weighted degree',
            'received issue comment',
            # 'received review',########
            'received review comment'
        ]
    if user_type != 'churner' and user_type != 'loyaler':
        print('User type error!')
        return
    if period_length == 120:
        step = 10
    elif period_length == 30:
        step = 5
    else:
        print('period length error!')
        return

    filenames = os.listdir(detailed_data_dir)
    data_filename = dict()
    for filename in filenames:
        if filename.split('_')[0][:-1] != user_type:
            continue
        if int(filename[0:-4].split('-')[1]) != period_length:
            continue
        if user_type=='loyaler' and float(filename[0:-4].split('-')[2]) != overlap_ratio:
            continue
        type_name = filename[filename.find('_')+1:filename.find('-')].replace('_',' ')
        if type_name in data_type_list:
            data_filename[type_name]=filename
    print(data_filename)#######################

    index_integrated_values = dict()
    for data_type in data_type_list:
        filename = data_filename[data_type]
        if user_type=='churner':
            filename2='loyalers_'+data_type.replace(' ','_')+'-'+str(period_length)+'-'+str(overlap_ratio)+'.csv'
        else:
            filename2 = 'churners_'+data_type.replace(' ','_')+'-'+str(period_length)+'.csv'
        max_value,min_value = getMaxMinValues(detailed_data_dir+'/'+filename,detailed_data_dir+'/'+filename2)
        print(filename,max_value,min_value)######################3

        with open(detailed_data_dir+'/'+filename,'r',encoding='utf-8')as f:
            f.readline()
            index = 0
            for line in f.readlines():
                index += 1
                items = line.strip(',\n').split(',')
                user_id = items[0]
                if index not in index_integrated_values.keys():
                    index_integrated_values[index]=[user_id]
                for j in range(1,len(items)):
                    value = (float(items[j]) - min_value) / (max_value - min_value)
                    index_integrated_values[index].append(value)
    for index in index_integrated_values.keys():
        print(index,len(index_integrated_values[index]),index_integrated_values[index])##################3

    save_filename = save_dir + '/' + user_type + '_' + 'integration' + '-' + str(period_length) + '-' + str(
        overlap_ratio) + '.csv'
    with open(save_filename, 'w', encoding='utf-8')as f:
        line = 'user_id,'
        for i in range(int(period_length / step) * len(data_type_list) + 2):
            line += str(i) + ','
        f.write(line + '\n')
        for index in index_integrated_values.keys():
            line = ''
            for value in index_integrated_values[index]:
                line+=str(value)+','
            if user_type=='churner':
                line+='1,'  # 正样本标签，流失开发者
            else:
                line += '-1,'  #负样本标签，非流失开发者
            f.write(line+'\n')
    f.close()


# 获取用于实际预测的整合的数据并存储
# id:仓库序号（1~30）
# period_length: 120或30
# 注：period_length和overlap_ratio共同用于确定文件版本号
# detailed_data_dir: 存储训练样本详细数据的文件夹
# train_max_min: 字典类型，训练集中不同类型数据的最大值和最小值，用于统一标准化
# data_type_list:整合的数据类型列表
def getIntegratedPredDataAndSave(detailed_data_dir, save_dir, period_length,train_max_min, data_type_list=None):
    if data_type_list is None:
        data_type_list = [
            'issue',
            'issue comment',
            'pull',
            'pull merged',
            'review comment',
            'betweeness',
            'weighted degree',
            'received issue comment',
            'received review comment'
        ]

    if period_length == 120:
        step = 10
    elif period_length == 30:
        step = 5
    else:
        print('period length error!')
        return

    filenames = os.listdir(detailed_data_dir)
    data_filename = dict()
    for filename in filenames:
        if int(filename[0:-4].split('-')[1]) != period_length:
            continue
        type_name = filename[filename.find('_')+1:filename.find('-')].replace('_',' ')
        if type_name in data_type_list:
            data_filename[type_name]=filename
    print(data_filename)#######################

    index_integrated_values = dict()
    for data_type in data_type_list:
        filename = data_filename[data_type]

        # 获取对应类型数据（包含训练集和待预测样本）的最大值和最小值
        max_value,min_value = getMaxMinValues(detailed_data_dir+'/'+filename)
        max_min = train_max_min[data_type]
        max_value,min_value = max(max_value,max_min[0]),min(min_value,max_min[1])  # ************

        print(filename,max_value,min_value)######################

        with open(detailed_data_dir+'/'+filename,'r',encoding='utf-8')as f:
            f.readline()
            index = 0
            for line in f.readlines():
                index += 1
                items = line.strip(',\n').split(',')
                user_id = items[0]
                if index not in index_integrated_values.keys():
                    index_integrated_values[index]=[user_id]
                for j in range(1,len(items)):
                    value = (float(items[j]) - min_value) / (max_value - min_value)
                    index_integrated_values[index].append(value)
    for index in index_integrated_values.keys():
        print(index,len(index_integrated_values[index]),index_integrated_values[index])##################3

    save_filename = save_dir + '/user_' + 'integration' + '-' + str(period_length) +'.csv'
    with open(save_filename, 'w', encoding='utf-8')as f:
        line = 'user_id,'
        for i in range(1,int(period_length / step) * len(data_type_list)+1):
            line += str(i) + ','
        f.write(line + '\n')
        for index in index_integrated_values.keys():
            line = ''
            for value in index_integrated_values[index]:
                line+=str(value)+','
            # if user_type=='churner':
            #     line+='1,'  # 正样本标签，流失开发者
            # else:
            #     line += '-1,'  #负样本标签，非流失开发者
            f.write(line+'\n')
    f.close()