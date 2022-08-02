from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from prediction_models.train_svm import getRandomIndex,getModelData
from prediction_models.result_analysis import get_proba_error,get_proba_metric
from prettytable import PrettyTable
from joblib import dump, load
import shutil
import os
import matplotlib.pyplot as plt
import matplotlib
import time
import pandas as pd
import random


def trainRF(split_balanced_data_dir,period_length=120,overlap_ratio=0.0,data_type_count=12,
            n_estimators=100,max_depth=None,min_samples_leaf=1,min_samples_split=2,max_features='sqrt',
            save_dir='rf_models',if_save=True,imp_metric='roc_auc'):
    train_data, test_data, train_label, test_label = getModelData(split_balanced_data_dir, period_length, overlap_ratio,
                                                                  data_type_count)
    model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_leaf=min_samples_leaf,
                                   min_samples_split=min_samples_split,max_features=max_features)
    model.fit(train_data, train_label)

    train_pred_proba = model.predict_proba(train_data)
    train_pred_proba = np.array([x[1] for x in train_pred_proba])
    test_pred_proba = model.predict_proba(test_data)
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
        model_filename = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + 'rf_model.joblib'
        dump(model, save_dir + '/' + model_filename)


# https://www.cnblogs.com/pinard/p/6160412.html
def gridSearchForRF(split_balanced_data_dir,period_length=120,overlap_ratio=0.0,data_type_count=12,
                    scoring='roc_auc',work_dir='.',save_dir='rf_models',if_save=True,imp_metric='roc_auc'):
    train_data, test_data, train_label, test_label = getModelData(split_balanced_data_dir, period_length, overlap_ratio,
                                                                  data_type_count)
    # 对n_estimators进行网格搜索
    param_test1 = {'n_estimators': np.linspace(100, 1000, 10, dtype=int)}
    gsearch1 = GridSearchCV(estimator=RandomForestClassifier(random_state=10),
                            param_grid=param_test1, scoring=scoring, verbose=2, refit=True, cv=5, n_jobs=-1)
    gsearch1.fit(train_data,train_label)
    best_estimators = gsearch1.best_params_['n_estimators']
    print('Best n_eatimators (approximately):',best_estimators)
    print(scoring+' score:',gsearch1.best_score_,'\n')

    param_test2 = {'n_estimators': np.linspace(best_estimators-50,best_estimators+50, 11, dtype=int)}
    gsearch2 = GridSearchCV(estimator=RandomForestClassifier(random_state=10),
                            param_grid=param_test2, scoring=scoring, verbose=2, refit=True, cv=5, n_jobs=-1)
    gsearch2.fit(train_data, train_label)
    best_estimators = gsearch2.best_params_['n_estimators']
    print('Best n_eatimators (accurately):', best_estimators)
    print(scoring+' score:', gsearch2.best_score_, '\n')

    # 对max_depth和min_samples_split进行网格搜索
    param_test3 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(2,31,2)}
    gsearch3 = GridSearchCV(estimator=RandomForestClassifier(random_state=10,n_estimators=best_estimators),
                            param_grid=param_test3, scoring=scoring, verbose=2, refit=True, cv=5, n_jobs=-1)
    gsearch3.fit(train_data, train_label)
    best_max_depth = gsearch3.best_params_['max_depth']
    best_min_samples_split = gsearch3.best_params_['min_samples_split']
    print('Best max_depth:',best_max_depth,'; min_samples_split:',best_min_samples_split)
    print(scoring+' score:', gsearch3.best_score_, '\n')

    # 对min_samples_split和min_samples_leaf进行网格搜索
    param_test4 = {'min_samples_split':range(max(2,best_min_samples_split-10),best_min_samples_split+11,2),
                    'min_samples_leaf':range(1,11,1)}
    gsearch4 = GridSearchCV(estimator=RandomForestClassifier(random_state=10,n_estimators=best_estimators,
                                                             max_depth=best_max_depth),
                            param_grid=param_test4, scoring=scoring, verbose=2, refit=True, cv=5, n_jobs=-1)
    gsearch4.fit(train_data, train_label)
    best_min_samples_split = gsearch4.best_params_['min_samples_split']
    best_min_samples_leaf = gsearch4.best_params_['min_samples_leaf']
    print('Best min_samples_split:',best_min_samples_split,'; best min_samples_leaf:',best_min_samples_leaf)
    print(scoring+' score:', gsearch4.best_score_, '\n')

    # 对max_features进行网格搜索
    param_test5 = {'max_features': range(2, 21, 2)}
    gsearch5 = GridSearchCV(estimator=RandomForestClassifier(random_state=10,n_estimators=best_estimators,
                                                             max_depth=best_max_depth,
                                                             min_samples_split=best_min_samples_split,
                                                             min_samples_leaf=best_min_samples_leaf),
                            param_grid=param_test5, scoring=scoring, verbose=2, refit=True, cv=5, n_jobs=-1)
    gsearch5.fit(train_data, train_label)
    best_max_features = gsearch5.best_params_['max_features']
    print('Best max_features:',best_max_features)
    print(scoring+' score:', gsearch5.best_score_, '\n')

    best_model = gsearch5.best_estimator_
    test_pred = best_model.predict(test_data)

    train_pred_proba = best_model.predict_proba(train_data)
    train_pred_proba = np.array([x[1] for x in train_pred_proba])
    test_pred_proba = best_model.predict_proba(test_data)
    test_pred_proba = np.array([x[1] for x in test_pred_proba])

    print('\n训练集误差结果：\n')
    train_error_results = get_proba_error(train_label, train_pred_proba, ['MSE', 'RMSE', 'MAE', 'SMAPE', 'R2'])
    print('\n测试集误差结果：\n')
    test_error_results = get_proba_error(test_label, test_pred_proba, ['MSE', 'RMSE', 'MAE', 'SMAPE', 'R2'])

    print('\n训练集不同指标结果：\n')
    train_metric_results = get_proba_metric(train_label, train_pred_proba, imp_metric=imp_metric)
    print('\n测试集不同指标结果：\n')
    test_metric_results = get_proba_metric(test_label, test_pred_proba, imp_metric=imp_metric)

    local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    tab1 = PrettyTable(['', 'MSE', 'RMSE', 'MAE', 'SMAPE', 'R2'])
    tab1.add_row(['train dataset', train_error_results['MSE'], train_error_results['RMSE'], train_error_results['MAE'],
                  train_error_results['SMAPE'], train_error_results['R2']])
    tab1.add_row(['test_dataset', test_error_results['MSE'], test_error_results['RMSE'], test_error_results['MAE'],
                  test_error_results['SMAPE'], test_error_results['R2']])
    tab2 = PrettyTable(['dataset(threshold)', 'accuracy', 'roc_auc', 'precision', 'recall', 'f1_score'])
    new_list = ['train dataset(' + "{0:.2f}".format(train_metric_results[0]) + ')']
    new_list.extend(train_metric_results[1])
    tab2.add_row(new_list)
    new_list = ['test dataset(' + "{0:.2f}".format(test_metric_results[0]) + ')']
    new_list.extend(test_metric_results[1])
    tab2.add_row(new_list)
    new_list = ['test dataset(0.5)']
    new_list.extend([accuracy_score(test_label, test_pred), roc_auc_score(test_label, test_pred),
                     precision_score(test_label, test_pred), recall_score(test_label, test_pred),
                     f1_score(test_label, test_pred)])
    tab2.add_row(new_list)

    ################################################################################3
    with open(work_dir+'/rf_result.csv', 'a', encoding='utf-8')as f:
        f.write('time: ' + local_time + '\n')
        tmp_index = split_balanced_data_dir.find('repo')
        f.write(split_balanced_data_dir[tmp_index:tmp_index + 7] + ',' +
                str(period_length) + ',' + str(overlap_ratio) + ',' + str(scoring) + ',\n')
        f.write(str(tab1) + '\n')
        f.write('用于确定最优阈值的指标为：' + imp_metric + '\n')
        f.write(str(tab2) + '\n')
        f.write('\n')
    #################################################################################
    with open(work_dir+'/model_params/rf_params.txt','w',encoding='utf-8')as f:
        f.write('time:' + local_time + '\n')
        f.write('n_estimators:'+str(best_estimators)+'\n')
        f.write('max_depth:'+str(best_max_depth)+'\n')
        f.write('min_samples_split:'+str(best_min_samples_split)+'\n')
        f.write('min_samples_leaf:'+str(best_min_samples_leaf)+'\n')
        f.write('max_features:'+str(best_max_features)+'\n')

    if if_save:
        s = 'Y'
    else:
        s = input('Do you want to save this model?[Y/n]')
    if s == 'Y' or s == 'y' or s == '':
        shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        model_filename = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + 'rf_best_model_'+scoring+\
                         '-'+str(period_length)+'-'+str(overlap_ratio)+'.joblib'
        dump(best_model, save_dir + '/' + model_filename)

    return best_estimators,best_max_depth,best_min_samples_split,best_min_samples_leaf,best_max_features


# 训练RF模型的统一接口，可以选择grid_search调参或直接训练
def train_rf(train_data_dir,repo_id,period_length=120,overlap_ratio=0.0,data_type_count=12,scoring='roc_auc',
             grid_search_control=False,model_params_dir='model_params',prediction_work_dir='.',
             save_dir='rf_models',if_save=True,imp_metric='roc_auc'):
    split_balanced_data_dir = train_data_dir + '/repo_' + str(repo_id) + '/split_balanced_data'
    if grid_search_control:
        gridSearchForRF(split_balanced_data_dir, period_length, overlap_ratio, data_type_count, scoring,
                        prediction_work_dir, save_dir, if_save,imp_metric)
    else:
        model_params_file = model_params_dir + '/rf_params.txt'
        params_dict = dict()
        with open(model_params_file, 'r', encoding='utf-8')as f:
            f.readline()
            for line in f.readlines():
                params_dict[line.split(':')[0]] = line.strip('\n').split(':')[1]
        n_estimators = int(params_dict['n_estimators'])
        max_depth = int(params_dict['max_depth'])
        min_samples_split = int(params_dict['min_samples_split'])
        min_samples_leaf = int(params_dict['min_samples_leaf'])
        max_features = int(params_dict['max_features'])

        trainRF(split_balanced_data_dir,period_length,overlap_ratio,data_type_count,n_estimators,max_depth,
                min_samples_leaf,min_samples_split,max_features,save_dir,if_save,imp_metric)