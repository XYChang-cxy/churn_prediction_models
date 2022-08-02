from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from joblib import dump, load
import shutil
import os
import matplotlib.pyplot as plt
import matplotlib
import time
import pandas as pd
import random
from prettytable import PrettyTable
from prediction_models.result_analysis import get_proba_metric,get_proba_error
from collections import Counter
np.random.seed(1)  # for reproducibility


def getRandomIndex(num,ran):
    list = random.sample(range(0,ran),num)
    return list


# 获取模型训练和验证的数据
# split_balanced_data_dir: 经过过采样处理得到的汇总文件
# period_length: 120或30
# overlap_ratio: 0.0或0.5，当user_type为loyaler时，获取数据时同一开发者不同区间的重叠度
# period_length和overlap_ratio共同用于确定使用的数据文件
# data_type_count:特征数量，默认12种特征
def getModelData(split_balanced_data_dir,period_length=120,overlap_ratio=0.0,data_type_count=12):
    if period_length == 120:
        col_count = 12*data_type_count
    elif period_length == 90:
        col_count = 9 * data_type_count
    elif period_length == 60:
        col_count = 6 * data_type_count
    elif period_length == 30:
        col_count = 6*data_type_count
    else:
        print('period length error!')
        return

    data_filename_1 = split_balanced_data_dir + '/balanced_data_train-' + str(period_length) + '-' + str(
        overlap_ratio) + '.csv'
    data_filename_2 = split_balanced_data_dir + '/balanced_data_test-' + str(period_length) + '-' + str(
        overlap_ratio) + '.csv'

    train_df = pd.read_csv(data_filename_1).iloc[:,1:col_count+2]
    train_array = np.array(train_df)
    np.random.shuffle(train_array)

    test_df = pd.read_csv(data_filename_2).iloc[:,1:col_count+2]
    test_array = np.array(test_df)
    np.random.shuffle(test_array)

    train_data,train_label = np.split(train_array,indices_or_sections=(-1,), axis=1)
    test_data,test_label = np.split(test_array,indices_or_sections=(-1,), axis=1)

    '''data_frame = pd.read_csv(data_filename).iloc[:,1:col_count+2]
    data_array = np.array(data_frame)
    np.random.shuffle(data_array)  # 打乱正负样本

    X, y = np.split(data_array, indices_or_sections=(-1,), axis=1)
    train_data, test_data, train_label, test_label = train_test_split(X, y, random_state=42, train_size=0.8,
                                                                      test_size=0.2)'''
    # 标签数组降维
    train_label = np.array([x[0] for x in train_label], dtype=int)
    test_label = np.array([x[0] for x in test_label], dtype=int)

    # print(train_data.shape,test_data.shape,Counter(train_label),Counter(test_label))
    return train_data,test_data,train_label,test_label


# SVM模型训练
# imp_metric:根据哪个指标来选取对预测结果（概率）进行二分类的最优阈值，可选参数包含：accuracy,roc_auc,precision,recall,f1_score
def trainSVM(split_balanced_data_dir,period_length=120,overlap_ratio=0.0,data_type_count=12,
             kernel='rbf',C=1.0,gamma='auto',degree=3,
             save_dir='svm_models',if_save=True,imp_metric='roc_auc'):
    train_data,test_data,train_label,test_label = getModelData(split_balanced_data_dir,period_length,overlap_ratio,
                                                               data_type_count)
    if kernel == 'rbf' or kernel == 'RBF':
        classifier = svm.SVC(kernel='rbf', C=C, gamma=gamma,probability=True)
    elif kernel == 'poly':
        classifier = svm.SVC(kernel='poly', C=C, degree=degree,probability=True)
    else:
        classifier = svm.SVC(kernel='linear', C=C,probability=True)

    classifier.fit(train_data, train_label)

    train_pred_proba = classifier.predict_proba(train_data)
    train_pred_proba = np.array([x[1] for x in train_pred_proba])
    test_pred_proba = classifier.predict_proba(test_data)
    test_pred_proba = np.array([x[1] for x in test_pred_proba])

    print('\n训练集误差结果：\n')
    get_proba_error(train_label, train_pred_proba, ['MSE', 'RMSE', 'MAE', 'SMAPE', 'R2'])
    print('\n测试集误差结果：\n')
    get_proba_error(test_label, test_pred_proba, ['MSE', 'RMSE', 'MAE', 'SMAPE', 'R2'])

    print('\n训练集不同指标结果：\n')
    get_proba_metric(train_label, train_pred_proba, imp_metric=imp_metric)
    print('\n测试集不同指标结果：\n')
    get_proba_metric(test_label, test_pred_proba, imp_metric=imp_metric)

    if if_save:
        shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        model_filename = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + 'svm_model.joblib'
        dump(classifier, save_dir + '/' + model_filename)


# GridSearch调参
# https://blog.csdn.net/aliceyangxi1987/article/details/73769950
# imp_metric:根据哪个指标来选取对预测结果（概率）进行二分类的最优阈值，可选参数包含：accuracy,roc_auc,precision,recall,f1_score
def gridSearchForSVM(split_balanced_data_dir,period_length=120,overlap_ratio=0.0,data_type_count=12,
                     scoring='roc_auc',work_dir='.',save_dir='svm_models',if_save=True,imp_metric='roc_auc'):
    train_data, test_data, train_label, test_label = getModelData(split_balanced_data_dir, period_length, overlap_ratio,
                                                                  data_type_count)
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 1, 5, 10],
                         'C': [0.1, 1, 10, 50, 100, 300, 500]},
                        {'kernel': ['linear'], 'C': [0.1, 1, 10, 50, 100, 300, 500]},
                        {'kernel': ['poly'], 'C': [0.1,1,10,50,100], 'degree': [2, 3, 4]}]
    scores = [scoring]
    best_params_dict = dict()
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
        clf = GridSearchCV(svm.SVC(probability=True), tuned_parameters, return_train_score=True,
                           scoring=score,verbose=2, refit=True, cv=5, n_jobs=-1)
        # 用训练集训练这个学习器 clf
        clf.fit(train_data, train_label)

        print("Best parameters set found on development set:")
        print()

        # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
        print(clf.best_params_)
        best_params_dict[score]=clf.best_params_

        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']

        # 看一下具体的参数间不同数值的组合后得到的分数是多少
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

        print()

        best_model = clf.best_estimator_
        test_pred = best_model.predict(test_data)

        train_pred_proba = best_model.predict_proba(train_data)
        train_pred_proba = np.array([x[1] for x in train_pred_proba])
        test_pred_proba = best_model.predict_proba(test_data)
        test_pred_proba = np.array([x[1] for x in test_pred_proba])

        print('\n训练集误差结果：\n')
        train_error_results = get_proba_error(train_label,train_pred_proba,['MSE','RMSE','MAE','SMAPE','R2'])
        print('\n测试集误差结果：\n')
        test_error_results = get_proba_error(test_label,test_pred_proba,['MSE','RMSE','MAE','SMAPE','R2'])

        print('\n训练集不同指标结果：\n')
        train_metric_results = get_proba_metric(train_label,train_pred_proba,imp_metric=imp_metric)
        print('\n测试集不同指标结果：\n')
        test_metric_results = get_proba_metric(test_label,test_pred_proba,imp_metric=imp_metric)

        local_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())

        tab1 = PrettyTable(['','MSE','RMSE','MAE','SMAPE','R2'])
        tab1.add_row(['train dataset',train_error_results['MSE'],train_error_results['RMSE'],train_error_results['MAE'],
                      train_error_results['SMAPE'],train_error_results['R2']])
        tab1.add_row(['test_dataset',test_error_results['MSE'],test_error_results['RMSE'],test_error_results['MAE'],
                      test_error_results['SMAPE'],test_error_results['R2']])
        tab2 = PrettyTable(['dataset(threshold)','accuracy','roc_auc','precision','recall','f1_score'])
        new_list = ['train dataset('+"{0:.2f}".format(train_metric_results[0])+')']
        new_list.extend(train_metric_results[1])
        tab2.add_row(new_list)
        new_list = ['test dataset(' + "{0:.2f}".format(test_metric_results[0]) + ')']
        new_list.extend(test_metric_results[1])
        tab2.add_row(new_list)
        new_list = ['test dataset(0.5)']
        new_list.extend([accuracy_score(test_label,test_pred),roc_auc_score(test_label,test_pred),
                         precision_score(test_label,test_pred),recall_score(test_label,test_pred),
                         f1_score(test_label,test_pred)])
        tab2.add_row(new_list)

        ################################################################################3
        with open(work_dir+'/svm_result.csv', 'a', encoding='utf-8')as f:
            f.write('time: ' + local_time + '\n')
            tmp_index = split_balanced_data_dir.find('repo')
            f.write(split_balanced_data_dir[tmp_index:tmp_index + 7] + ',' +
                    str(period_length) + ',' + str(overlap_ratio) + ',' + str(scoring) + ',\n')
            f.write(str(tab1)+'\n')
            f.write('用于确定最优阈值的指标为：'+imp_metric+'\n')
            f.write(str(tab2)+'\n')
            f.write('\n')
        #################################################################################
        with open(work_dir+'/model_params/svm_params.txt','w',encoding='utf-8')as f:
            f.write('time:' + local_time + '\n')
            for key in best_params_dict[scoring].keys():
                f.write(str(key)+':'+str(best_params_dict[scoring][key])+'\n')

        if if_save:
            s = 'Y'
        else:
            s = input('Do you want to save this model?[Y/n]')
        if s == 'Y' or s == 'y' or s == '':
            shutil.rmtree(save_dir)
            os.mkdir(save_dir)
            model_filename = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + 'svm_best_model_'+score+\
                             '-'+str(period_length)+'-'+str(overlap_ratio)+'.joblib'
            dump(best_model, save_dir + '/' + model_filename)

    return best_params_dict


# 训练SVM模型的统一接口，可以选择grid_search调参或直接训练
def train_svm(train_data_dir,repo_id,period_length=120,overlap_ratio=0.0,data_type_count=12,scoring='roc_auc',
              grid_search_control=False,model_params_dir='model_params',prediction_work_dir='.',
              save_dir='svm_models',if_save=True,imp_metric='roc_auc'):
    split_balanced_data_dir = train_data_dir+'/repo_'+str(repo_id)+'/split_balanced_data'
    if grid_search_control:
        gridSearchForSVM(split_balanced_data_dir,period_length,overlap_ratio,data_type_count,scoring,
                         prediction_work_dir,save_dir,if_save,imp_metric)
    else:
        model_params_file = model_params_dir + '/svm_params.txt'
        params_dict=dict()
        with open(model_params_file,'r',encoding='utf-8')as f:
            f.readline()
            for line in f.readlines():
                params_dict[line.split(':')[0]]=line.strip('\n').split(':')[1]
        kernel = params_dict['kernel']
        C = float(params_dict['C'])
        if kernel == 'linear':
            trainSVM(split_balanced_data_dir, period_length, overlap_ratio, data_type_count, kernel, C,
                     save_dir=save_dir,if_save=if_save,imp_metric=imp_metric)
        elif kernel == 'rbf':
            gamma = float(params_dict['gamma'])
            trainSVM(split_balanced_data_dir, period_length, overlap_ratio, data_type_count, kernel, C, gamma,
                     save_dir=save_dir, if_save=if_save,imp_metric=imp_metric)
        else:
            degree = int(params_dict['degree'])
            trainSVM(split_balanced_data_dir, period_length, overlap_ratio, data_type_count, kernel, C,
                     degree=degree, save_dir=save_dir, if_save=if_save,imp_metric=imp_metric)


if __name__ == '__main__':
    split_balanced_data_dir = r'F:\MOOSE_cxy\developer_churn_prediction\churn_prediction\data_preprocess\data\repo_2\part_all\balanced_data'
    period_length = 120
    overlap_ratio = 0.0
    data_type_count = 12
    # gridSearchForSVM(split_balanced_data_dir,period_length,overlap_ratio,data_type_count)
    # trainSVM(split_balanced_data_dir,period_length,overlap_ratio,data_type_count)
    # train_svm('../data_preprocess/train_data',8649239,120,0.0,9,grid_search_control=False)
