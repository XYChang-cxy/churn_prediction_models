import datetime
import numpy as np
import pandas as pd
import os
import shutil
from joblib import dump,load
from collections import Counter
from data_preprocess.database_connect import *
from data_preprocess.get_user import saveUserActivePeriod,getModelUserPeriod,getPredictionUserPeriod
from data_preprocess.get_detailed_data import getCountDataAndSave,getDCNDataAndSave,getReceivedDataAndSave
from data_preprocess.get_integrated_data import getIntegratedDataAndSave,getIntegratedPredDataAndSave
from data_preprocess.get_balanced_integrated_data import getSplitBanlancedDataAndSave
from data_preprocess.get_max_min_values import getMaxMinValues

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

'''train_data_dir = 'train_data'
prediction_data_dir = 'prediction_data' '''


# 获取用于模型训练的数据，该函数会在train_data文件夹创建相应数据文件，无返回值
# repo_id: 仓库id
# train_data_dir: 存储用于训练模型的数据的文件夹
# period_length: 输入数据的时间跨度，目前仅支持120天或30天
# overlap_ratio: 负样本（忠诚开发者数据）采样区间重合度，默认为0.0,即不重合
# churn_limit_weeks: 流失期限，默认14周
# train_end_time: 用于获取训练数据的截止时间，（开始时间是仓库创建时间）
# continue_runing: 是否在处理数据过程中不间断运行，默认为True
# time_threshold: 用于划分开发者的百分位数，若大于0小于1表示对应百分位数；若为整数则表示具体天数。默认为0.8，即剔除活动时间少于第80百分位数的开发者
# 返回值：可直接输入模型训练的数据存储路径；筛选开发者的阈值（单位是天）
def train_data_preprocess(repo_id,train_data_dir,period_length=120,overlap_ratio=0.0,churn_limit_weeks=14,
                          train_end_time = '2022-01-01',continue_running=False,
                          time_threshold=0.8):
    if 0 < time_threshold < 1:
        time_threshold_percentile = int(time_threshold * 100)
    else:
        time_threshold_percentile = 80

    user_type_list = ['churner', 'loyaler']

    repo_info = getRepoInfoFromTable(repo_id,['created_at'])
    create_time = repo_info[0][0:10]

    print('**************************Data Preprocess**************************')
    print('\nStep1: make directories.')
    # ① 为目标仓库创建存储数据的文件夹
    repo_data_dir = train_data_dir+'/repo_'+str(repo_id)
    if not os.path.exists(repo_data_dir+'/detailed_data'):
        os.makedirs(repo_data_dir+'/detailed_data')
    if not os.path.exists(repo_data_dir+'/normalized_data'):
        os.makedirs(repo_data_dir+'/normalized_data')
    if not os.path.exists(repo_data_dir+'/split_balanced_data'):
        os.makedirs(repo_data_dir+'/split_balanced_data')


    if continue_running:
        s = 'Y'
    else:
        s = input('Step1 has finished, continue?[Y/n]')
    if s!='Y' and s!='y' and s!='':
        return
    print('\nStep2: get active period for all users.')
    # ② 获取所有流失用户和留存用户的连续活动时间（第一次活动和最后一次活动时间）
    time_threshold_days = int(
        saveUserActivePeriod(repo_id, create_time, train_end_time, repo_data_dir, churn_limit_weeks,
                             time_threshold_percentile))

    if continue_running:
        s = 'Y'
    else:
        s = input('Step2 has finished, continue?[Y/n]')
    if s != 'Y' and s != 'y' and s != '':
        return
    print('\nStep3: get users and correspond period for model training.')
    # ③ 获取剔除不重要开发者（活动时间少于第90百分位数）后，分成churner和loyaler两部分，并生成重要开发者的取样区间
    for user_type in user_type_list:
        if 0 < time_threshold < 1:
            getModelUserPeriod(repo_id, user_type, repo_data_dir,
                               repo_data_dir + '/' + str(repo_id) + '_user_active_period.csv',
                               churn_limit_weeks, period_length, overlap_ratio, time_threshold=-1)
        else:
            getModelUserPeriod(repo_id, user_type, repo_data_dir,
                               repo_data_dir + '/' + str(repo_id) + '_user_active_period.csv',
                               churn_limit_weeks, period_length, overlap_ratio, time_threshold=time_threshold)

    if continue_running:
        s = 'Y'
    else:
        s = input('Step3 has finished, continue?[Y/n]')
    if s != 'Y' and s != 'y' and s != '':
        return
    print('\nStep4: get detailed sample data for model training.')
    # ④ 根据选取的开发者和区域区间，获取样本详细数据，并存储到detailed_data文件夹
    for user_type in user_type_list:
        if user_type == 'churner':
            period_filename = 'repo_' + user_type + 's_period-' + str(period_length) + '.csv'
        else:
            period_filename = 'repo_' + user_type + 's_period-' + str(period_length) + '-' + str(
                overlap_ratio) + '.csv'
        for data_type in data_type_list:
            if data_type == 'betweeness' or data_type == 'weighted degree':
                getDCNDataAndSave(repo_id, repo_data_dir + '/' + period_filename, period_length, data_type,
                                  repo_data_dir + '/detailed_data')
            elif data_type.find('received') != -1:
                getReceivedDataAndSave(repo_id, repo_data_dir + '/' + period_filename, period_length, data_type[9:],
                                       repo_data_dir + '/detailed_data')
            else:
                getCountDataAndSave(repo_id, repo_data_dir + '/' + period_filename, period_length, data_type,
                                    repo_data_dir + '/detailed_data')

    if continue_running:
        s = 'Y'
    else:
        s = input('Step4 has finished, continue?[Y/n]')
    if s != 'Y' and s != 'y' and s != '':
        return
    print('\nStep5: get integrated and normalized data.')
    # ⑤ 根据详细数据生成整合的标准化后的数据
    for user_type in user_type_list:
        getIntegratedDataAndSave(repo_data_dir + '/detailed_data', repo_data_dir + '/normalized_data',
                                 user_type, period_length, overlap_ratio, data_type_list)

    if continue_running:
        s = 'Y'
    else:
        s = input('Step5 has finished, continue?[Y/n]')
    if s != 'Y' and s != 'y' and s != '':
        return
    print('\nStep6: get balanced data.')
    getSplitBanlancedDataAndSave(repo_data_dir + '/normalized_data', repo_data_dir + '/split_balanced_data',
                                    period_length, overlap_ratio, data_type_list,1,split_ratio=0.8)
    print('Data preprocessing finished.')
    return time_threshold_days


# 获取近期需要预测的开发者的数据，该函数会在prediction_data文件夹创建相应数据文件，并返回可输入模型的数据
# repo_id: 仓库id
# train_data_dir: 存储用于训练模型的数据的文件夹
# prediction_data_dir: 存储用于模型预测的数据的文件夹
# period_length: 输入数据的时间跨度，目前仅支持120天或30天
# churn_limit_weeks: 流失期限，默认14周
# time_threshold_days: 剔除临时开发者的活动时间阈值，默认28天
# continue_runing: 是否在处理数据过程中不间断运行，默认为True
# 返回值：需要预测的user_id列表，和对应的模型输入数据列表
def prediction_data_preprocess(repo_id,train_data_dir,prediction_data_dir,period_length=120,churn_limit_weeks=14,time_threshold_days=28,
                               continue_running=True):

    repo_info = getRepoInfoFromTable(repo_id,['created_at'])
    create_time = repo_info[0][0:10]
    end_time = datetime.datetime.now().strftime(fmt_day)

    churn_limit_days = churn_limit_weeks*7  # 流失期限，单位是天

    # 获取数据的开始时间,即 “流失期限天数+输入数据天数” 前
    start_time = (datetime.datetime.now()-datetime.timedelta(days=churn_limit_days+period_length)).strftime(fmt_day)
    if start_time < create_time:
        start_time = create_time

    print('**************************Data Preprocess For Prediction**************************')
    print('\nStep1: make directories.')
    # ① 为目标仓库创建存储数据的文件夹
    repo_data_dir = prediction_data_dir+'/repo_'+str(repo_id)
    # shutil.rmtree(repo_data_dir)  # 获取新数据前先将旧数据删除
    if not os.path.exists(repo_data_dir+'/detailed_data'):
        os.makedirs(repo_data_dir+'/detailed_data')
    if not os.path.exists(repo_data_dir+'/normalized_data'):
        os.makedirs(repo_data_dir+'/normalized_data')

    if continue_running:
        s = 'Y'
    else:
        s = input('Step1 has finished, continue?[Y/n]')
    if s!='Y' and s!='y' and s!='':
        return
    print('\nStep2: get active period for all users.')
    # ② 获取近一段时间（流失期限+最大观测区间长度）的所有开发者活跃时间段
    saveUserActivePeriod(repo_id,start_time,end_time,repo_data_dir,churn_limit_weeks)

    if continue_running:
        s = 'Y'
    else:
        s = input('Step2 has finished, continue?[Y/n]')
    if s != 'Y' and s != 'y' and s != '':
        return
    print('\nStep3: get users and correspond period for model training.')
    # ③ 获取剔除不重要开发者（活动时间少于第90百分位数）后，分成churner和loyaler两部分，并生成重要开发者的取样区间
    getPredictionUserPeriod(repo_id,repo_data_dir,repo_data_dir+'/'+str(repo_id)+'_user_active_period.csv',period_length,
                            time_threshold_days)

    if continue_running:
        s = 'Y'
    else:
        s = input('Step3 has finished, continue?[Y/n]')
    if s != 'Y' and s != 'y' and s != '':
        return
    print('\nStep4: get detailed sample data for model training.')
    # ④ 根据选取的开发者和区域区间，获取样本详细数据，并存储到detailed_data文件夹
    period_filename = 'repo_users_period-' + str(period_length) + '.csv'
    for data_type in data_type_list:
        if data_type == 'betweeness' or data_type == 'weighted degree':
            getDCNDataAndSave(repo_id, repo_data_dir + '/' + period_filename, period_length, data_type,
                              repo_data_dir + '/detailed_data')
        elif data_type.find('received') != -1:
            getReceivedDataAndSave(repo_id, repo_data_dir + '/' + period_filename, period_length, data_type[9:],
                                   repo_data_dir + '/detailed_data')
        else:
            getCountDataAndSave(repo_id, repo_data_dir + '/' + period_filename, period_length, data_type,
                                repo_data_dir + '/detailed_data')

    if continue_running:
        s = 'Y'
    else:
        s = input('Step4 has finished, continue?[Y/n]')
    if s != 'Y' and s != 'y' and s != '':
        return
    print('\nStep5: get integrated and normalized data.')
    # ⑤ 根据详细数据生成整合的标准化后的数据
    train_detailed_dir = train_data_dir + '/repo_' + str(repo_id) + '/detailed_data'
    if not os.path.exists(train_detailed_dir):######################### 后续完成模型训练后需要删除
        print('\"train_data\" not exist, use \"data\" to get max/min values.')
        train_detailed_dir = 'data/repo_20/part_all/detailed_data'
    train_max_min = getTrainMaxMin(train_detailed_dir,period_length)  # 训练集中不同类型数据对应的最大值和最小值
    print(train_max_min)
    getIntegratedPredDataAndSave(repo_data_dir+'/detailed_data',repo_data_dir+'/normalized_data',period_length,
                                 train_max_min,data_type_list)

    print('Data preprocessing finished.')
    return get_existed_prediction_data(repo_data_dir+'/normalized_data')


# 获取训练集各类数据的最大值和最小值
# 返回值为字典：data_type:[max_val,min_val]
def getTrainMaxMin(train_detailed_dir,period_length,overlap_ratio=0.0):
    filenames = os.listdir(train_detailed_dir)
    data_filename = dict()
    for filename in filenames:
        if filename.split('_')[0][:-1] != 'churner':
            continue
        if int(filename[0:-4].split('-')[1]) != period_length:
            continue
        type_name = filename[filename.find('_') + 1:filename.find('-')].replace('_', ' ')
        if type_name in data_type_list:
            data_filename[type_name] = filename
    train_max_min = dict()
    for data_type in data_type_list:
        filename = data_filename[data_type]
        filename2 = 'loyalers_' + data_type.replace(' ', '_') + '-' + str(period_length) + '-' + str(
            overlap_ratio) + '.csv'
        max_val,min_val = getMaxMinValues(train_detailed_dir+'/'+filename,train_detailed_dir+'/'+filename2)
        train_max_min[data_type]=[max_val,min_val]
    return train_max_min


# 根据预处理生成的预测数据文件，获取模型输入
# file_path: 存储生成文件的文件夹（必须保证该路径下仅有一个文件），或具体的文件路径
def get_existed_prediction_data(file_path):
    if file_path.find('.csv')==-1:
        filenames = os.listdir(file_path)
        if len(filenames)==1:
            file_path = file_path+'/'+filenames[0]
        else:
            print('Data file is ambiguous or not found! Please check the file_path!')
    period_length = int(file_path[:-4].split('-')[-1])
    if period_length == 120:
        col_count = 12 * len(data_type_list)
    elif period_length == 30:
        col_count = 6 * len(data_type_list)
    else:
        print('period length error!')
        return

    user_id_list = np.array(pd.read_csv(file_path).iloc[:,0])
    input_data = np.array(pd.read_csv(file_path).iloc[:, 1:col_count+1])
    return user_id_list,input_data


if __name__ == '__main__':
    repo_id = 8649239
    period_length=120
    overlap_ratio = 0.0
    churn_limit_weeks = 14
    time_threshold_days = 28
    train_end_time = '2022-01-01'
    train_data_dir = 'train_data'
    prediction_data_dir = 'prediction_data'

    # 测试一：生成预测数据集，并返回
    user_id_list,input_data = prediction_data_preprocess(repo_id,train_data_dir,prediction_data_dir,
                                                         period_length,churn_limit_weeks,time_threshold_days)
    print(user_id_list)
    print(input_data)

    prediction_file = prediction_data_dir+'/repo_'+str(repo_id)+'/normalized_data'
    # 测试二：根据生成的预测数据集文件，直接返回数据
    # user_id_list,input_data = get_existed_prediction_data(prediction_file)
    # print(len(user_id_list))
    # print(input_data.shape)

    # 测试三：加载训练好的模型，对input_data进行预测
    '''model_path = '../prediction_models/xgboost_models/2022-06-03_15-16-04xgboost_best_model_roc_auc-120-0.0.joblib'
    # model_path = '../prediction_models/rf_models/2022-06-03_15-11-48rf_best_model_roc_auc-120-0.0.joblib'
    # model_path = '../prediction_models/adaboost_models/2022-06-03_15-15-13adaboost_best_model_roc_auc-120-0.0.joblib'
    # model_path = '../prediction_models/svm_models/2022-06-03_15-06-53svm_best_model_roc_auc-120-0.0.joblib'
    model = load(model_path)
    y_pred = model.predict(input_data)
    for i in range(len(user_id_list)):
        print(user_id_list[i],'\t\t',y_pred[i])
    print(Counter(y_pred))'''

    # 测试四：训练集数据预处理
    # train_data_preprocess(repo_id,train_data_dir,period_length,overlap_ratio,churn_limit_weeks,train_end_time,False,
    #                       time_threshold=time_threshold_days)
