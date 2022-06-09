# 该文件用于将标准化处理后的整合数据进行SMOTE过采样，以达到平衡正负样本的目的
import numpy as np
import pandas as pd
import os
import random
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
np.random.seed(1)  # for reproducibility

def getRandomIndex(num,ran):
    list = random.sample(range(0,ran),num)
    return list


# 根据标准化后的正负样本的整合文件，先划分训练集和测试集，然后对训练集进行SMOTE过采样
# 注：直接对所有数据进行过采样，没有先划分训练集和测试集，该方法被舍弃
# period_length: 120或30
# overlap_ratio: 当user_type为loyaler时，获取数据时同一开发者不同区间的重叠度
# split_mode:划分训练集和测试集的模式；0--使用train_test_split划分；1--严格按照比例划分
def getSplitBanlancedDataAndSave(normalized_data_dir,save_dir,period_length, overlap_ratio,
                                 data_type_list=None,split_mode=1,split_ratio=0.8):
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
        col_count = 12*len(data_type_list)
    elif period_length == 30:
        col_count = 6*len(data_type_list)
    else:
        print('period length error!')
        return
    filename1 = normalized_data_dir+'/'+'churner_integration-'+str(period_length)+'-'+str(overlap_ratio)+'.csv'
    filename2 = normalized_data_dir + '/' + 'loyaler_integration-' + str(period_length) + '-' + str(
        overlap_ratio) + '.csv'

    content1 = pd.read_csv(filename1).iloc[:,:col_count+2]
    content2 = pd.read_csv(filename2).iloc[:,:col_count+2]

    all_zero_row_index = []
    for i in range(content2.shape[0]):
        if np.max(np.array(content2.iloc[i,1:]))==0.0:
            # print(i,'\n',content2.iloc[i],'\n')
            all_zero_row_index.append(i)
    print('Drop '+str(len(all_zero_row_index))+' all-zero rows in loyaler data.')

    content2.drop(all_zero_row_index,axis=0,inplace=True)  # inplace=True 在原数据上更改


    if split_mode==0:# 方式1：使用train_test_split随机划分训练集和测试集（训练集和测试集正负样本比例不完全一样）
        # 混合正负样本(未打乱顺序）
        all_data = content1.append(content2)
        all_data = np.array(all_data)
        np.random.shuffle(all_data)
        # 划分特征和标签
        user_id_list,X,y = np.split(all_data,indices_or_sections=(1,-1,),axis=1)

        train_data, test_data, train_label, test_label = train_test_split(X, y, random_state=42, train_size=split_ratio,
                                                                          test_size=1-split_ratio)
        train_label = np.array([x[0] for x in train_label], dtype=int)
        test_label = np.array([x[0] for x in test_label], dtype=int)
    else:
        array1 = np.array(content1)
        array2 = np.array(content2)

        random_index_1 = getRandomIndex(int(array1.shape[0]*split_ratio),array1.shape[0])
        train_churn = np.array([array1[i] for i in random_index_1])
        test_churn = np.array([array1[i] for i in range(array1.shape[0]) if i not in random_index_1])

        random_index_2 = getRandomIndex(int(array2.shape[0]*split_ratio),array2.shape[0])
        train_loyal = np.array([array2[i] for i in random_index_2])
        test_loyal = np.array([array2[i] for i in range(array2.shape[0]) if i not in random_index_2])

        train_array = np.append(train_churn,train_loyal,axis=0)
        test_array = np.append(test_churn,test_loyal,axis=0)

        np.random.shuffle(train_array)
        np.random.shuffle(test_array)
        # print(train_churn.shape,test_churn.shape)
        # print(train_loyal.shape,test_loyal.shape)
        # print(train_array.shape,test_array.shape)
        train_user_id_list,train_data,train_label = np.split(train_array,indices_or_sections=(1,-1,),axis=1)
        test_user_id_list,test_data,test_label = np.split(test_array,indices_or_sections=(1,-1,),axis=1)
        train_label = np.array([x[0] for x in train_label], dtype=int)
        test_label = np.array([x[0] for x in test_label], dtype=int)

    # 降维
    # user_id_list=np.array([x[0] for x in user_id_list], dtype=int)

    smote = SMOTE(random_state=42)
    train_data_smote,train_label_smote = smote.fit_resample(train_data,train_label)

    output_df = pd.concat([pd.DataFrame(train_data_smote),pd.DataFrame(train_label_smote,columns=['label'])],axis=1)
    output_df = output_df.sample(frac=1,random_state=42).reset_index(drop=True)
    output_filename = save_dir+'/balanced_data_train-'+str(period_length)+'-'+str(overlap_ratio)+'.csv'
    output_df.to_csv(output_filename)
    output_df = pd.concat([pd.DataFrame(test_data),pd.DataFrame(test_label,columns=['label'])],axis=1)
    output_filename = save_dir + '/balanced_data_test-' + str(period_length) + '-' + str(overlap_ratio) + '.csv'
    output_df.to_csv(output_filename)
    print(Counter(train_label),Counter(train_label_smote),Counter(test_label))


if __name__ == '__main__':
    getSplitBanlancedDataAndSave(r'F:\MOOSE_cxy\developer_churn_prediction\churn_prediction\data_preprocess\data\repo_2\part_all\normalized_data',
                             r'C:\Users\cxy\Desktop\test',30,0.0)