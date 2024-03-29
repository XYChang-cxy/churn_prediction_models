import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from prediction_models.train_svm import getRandomIndex,getModelData
from prediction_models.result_analysis import get_proba_error,get_proba_metric
from prettytable import PrettyTable
from joblib import dump, load
import shutil
import os
import matplotlib.pyplot as plt
import matplotlib
import xgboost as xgb
import time
import pandas as pd
import random


def trainXGBoost(split_balanced_data_dir,period_length=120,overlap_ratio=0.0,data_type_count=12,other_params=None,
                 n_estimators=500,max_depth=6,min_child_weight=1,subsample=1,colsample_bytree=1,
                 gamma=0,reg_lambda=1,reg_alpha=0,eta=0.3,
                 save_dir='xgboost_models',if_save=True,imp_metric='roc_auc'):
    train_data, test_data, train_label, test_label = getModelData(split_balanced_data_dir, period_length, overlap_ratio,
                                                                  data_type_count)
    for i in range(train_label.shape[0]):
        if train_label[i]==-1:
            train_label[i]=0
    for i in range(test_label.shape[0]):
        if test_label[i]==-1:
            test_label[i]=0

    if other_params == None:
        other_params = {
            'eta': eta,
            'n_estimators': n_estimators,
            'gamma': gamma,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'colsample_bytree': colsample_bytree,
            'subsample': subsample,
            'reg_lambda': reg_lambda,
            'reg_alpha': reg_alpha,
            'seed': 33
        }
    model = xgb.XGBClassifier(**other_params)
    model.fit(train_data,train_label)
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
        model_filename = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + 'xgboost_model.joblib'
        dump(model, save_dir + '/' + model_filename)


def gridSearchForXGBoost(split_balanced_data_dir,period_length=120,overlap_ratio=0.0,data_type_count=12,
                         work_dir='.',save_dir='xgboost_models',scoring='roc_auc',if_save=True,imp_metric='roc_auc'):
    train_data, test_data, train_label, test_label = getModelData(split_balanced_data_dir, period_length, overlap_ratio,
                                                                  data_type_count)
    for i in range(train_label.shape[0]):
        if train_label[i]==-1:
            train_label[i]=0
    for i in range(test_label.shape[0]):
        if test_label[i]==-1:
            test_label[i]=0
    other_params = {
        'eta': 0.3,
        'n_estimators': 500,
        'gamma': 0,
        'max_depth': 6,
        'min_child_weight': 1,
        'colsample_bytree': 1,
        'colsample_bylevel': 1,
        'subsample': 1,
        'reg_lambda': 1,
        'reg_alpha': 0,
        'seed': 33
    }
    model = xgb.XGBClassifier(**other_params)

    # 1. 确定n_estimators
    cv_params = {'n_estimators': np.linspace(100, 1000, 10, dtype=int)}
    gs = GridSearchCV(model,cv_params,scoring=scoring,verbose=2, refit=True, cv=5, n_jobs=-1)
    gs.fit(train_data,train_label)
    n_estimators = gs.best_params_['n_estimators']
    print('1. Best n_estimators(approximately):',n_estimators)
    print(scoring+' score:',gs.best_score_)

    cv_params = {'n_estimators': np.linspace(n_estimators-50,n_estimators+50, 11, dtype=int)}
    gs = GridSearchCV(model, cv_params, scoring=scoring, verbose=2, refit=True, cv=5, n_jobs=-1)
    gs.fit(train_data, train_label)
    n_estimators = gs.best_params_['n_estimators']
    print('1. Best n_estimators(accurately):', n_estimators)
    print(scoring+' score:', gs.best_score_)

    other_params['n_estimators']=n_estimators
    model = xgb.XGBClassifier(**other_params)

    # 2. 确定max_depth
    cv_params = {'max_depth': np.linspace(1, 10, 11, dtype=int)}
    gs = GridSearchCV(model, cv_params, scoring=scoring, verbose=2, refit=True, cv=5, n_jobs=-1)
    gs.fit(train_data, train_label)
    max_depth = gs.best_params_['max_depth']
    print('2. Best max_depth:', max_depth)
    print(scoring+' score:', gs.best_score_)

    other_params['max_depth']=max_depth
    model = xgb.XGBClassifier(**other_params)

    # 3. 确定min_child_weight
    cv_params = {'min_child_weight': np.linspace(1, 10, 11, dtype=int)}
    gs = GridSearchCV(model, cv_params, scoring=scoring, verbose=2, refit=True, cv=5, n_jobs=-1)
    gs.fit(train_data, train_label)
    min_child_weight = gs.best_params_['min_child_weight']
    print('3. Best min_child_weight:', min_child_weight)
    print(scoring+' score:', gs.best_score_)

    other_params['min_child_weight'] = min_child_weight
    model = xgb.XGBClassifier(**other_params)

    # 4. 确定gamma
    cv_params = {'gamma': np.linspace(0, 1, 11)}
    gs = GridSearchCV(model, cv_params, scoring=scoring, verbose=2, refit=True, cv=5, n_jobs=-1)
    gs.fit(train_data, train_label)
    gamma = gs.best_params_['gamma']
    print('4. Best gamma(approximately):', gamma)
    print(scoring+' score:', gs.best_score_)

    if gamma==0:
        cv_params = {'gamma': np.linspace(0, 0.1, 11)}
    else:
        cv_params = {'gamma': np.linspace(gamma-0.05, gamma+0.05, 11)}
    gs = GridSearchCV(model, cv_params, scoring=scoring, verbose=2, refit=True, cv=5, n_jobs=-1)
    gs.fit(train_data, train_label)
    gamma = gs.best_params_['gamma']
    print('4. Best gamma(accurately):', gamma)
    print(scoring+' score:', gs.best_score_)

    other_params['gamma'] = gamma
    model = xgb.XGBClassifier(**other_params)

    # 5. 确定subsample
    cv_params = {'subsample': np.linspace(0.1, 1, 10)}
    gs = GridSearchCV(model, cv_params, scoring=scoring, verbose=2, refit=True, cv=5, n_jobs=-1)
    gs.fit(train_data, train_label)
    subsample = gs.best_params_['subsample']
    print('5. Best subsample(approximately):', subsample)
    print(scoring+' score:', gs.best_score_)

    if subsample==1:
        cv_params = {'subsample': np.linspace(0.9, 1, 11)}
    else:
        cv_params = {'subsample': np.linspace(subsample-0.05, subsample+0.05, 11)}
    gs = GridSearchCV(model, cv_params, scoring=scoring, verbose=2, refit=True, cv=5, n_jobs=-1)
    gs.fit(train_data, train_label)
    subsample = gs.best_params_['subsample']
    print('5. Best subsample(accurately):', subsample)
    print(scoring+' score:', gs.best_score_)

    other_params['subsample'] = subsample
    model = xgb.XGBClassifier(**other_params)

    # 6.确定colsample_bytree
    cv_params = {'colsample_bytree': np.linspace(0, 1, 11)[1:]}
    gs = GridSearchCV(model, cv_params, scoring=scoring, verbose=2, refit=True, cv=5, n_jobs=-1)
    gs.fit(train_data, train_label)
    colsample_bytree = gs.best_params_['colsample_bytree']
    print('6. Best colsample_bytree:', colsample_bytree)
    print(scoring+' score:', gs.best_score_)

    other_params['colsample_bytree'] = colsample_bytree
    model = xgb.XGBClassifier(**other_params)

    # 7.确定reg_lambda
    cv_params = {'reg_lambda': np.linspace(0, 100, 11)}
    gs = GridSearchCV(model, cv_params, scoring=scoring, verbose=2, refit=True, cv=5, n_jobs=-1)
    gs.fit(train_data, train_label)
    reg_lambda = gs.best_params_['reg_lambda']
    print('7. Best reg_lambda(approximately):', reg_lambda)
    print(scoring+' score:', gs.best_score_)

    if reg_lambda == 0:
        cv_params = {'reg_lambda': np.linspace(0, 10, 11)}
    else:
        cv_params = {'reg_lambda': np.linspace(reg_lambda-10, reg_lambda+10, 11)}
    gs = GridSearchCV(model, cv_params, scoring=scoring, verbose=2, refit=True, cv=5, n_jobs=-1)
    gs.fit(train_data, train_label)
    reg_lambda = gs.best_params_['reg_lambda']
    print('7. Best reg_lambda(accurately):', reg_lambda)
    print(scoring+' score:', gs.best_score_)

    other_params['reg_lambda'] = reg_lambda
    model = xgb.XGBClassifier(**other_params)

    # 8. 确定reg_alpha
    cv_params = {'reg_alpha': np.linspace(0, 10, 11)}
    gs = GridSearchCV(model, cv_params, scoring=scoring, verbose=2, refit=True, cv=5, n_jobs=-1)
    gs.fit(train_data, train_label)
    reg_alpha = gs.best_params_['reg_alpha']
    print('8. Best reg_alpha:', reg_alpha)
    print(scoring+' score:', gs.best_score_)

    other_params['reg_alpha'] = reg_alpha
    model = xgb.XGBClassifier(**other_params)

    # 9.确定eta
    cv_params = {'eta':np.array([0.001,0.002,0.005,0.01, 0.02, 0.05, 0.1, 0.15,0.3,0.5])}
    # cv_params = {'eta': np.logspace(-2, 0, 10)}
    gs = GridSearchCV(model, cv_params, scoring=scoring, verbose=2, refit=True, cv=5, n_jobs=-1)
    gs.fit(train_data, train_label)
    eta = gs.best_params_['eta']
    print('9. Best eta:', eta)
    print(scoring+' score:', gs.best_score_)

    other_params['eta'] = eta

    best_model = gs.best_estimator_
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
    with open(work_dir+'/xgboost_result.csv','a',encoding='utf-8')as f:
        f.write('time: ' + local_time + '\n')
        tmp_index = split_balanced_data_dir.find('repo')
        f.write(split_balanced_data_dir[tmp_index:tmp_index + 7] + ',' +
                str(period_length) + ',' + str(overlap_ratio) + ',' + str(scoring) + ',\n')
        f.write(str(tab1) + '\n')
        f.write('用于确定最优阈值的指标为：' + imp_metric + '\n')
        f.write(str(tab2) + '\n')
        f.write('\n')
    #################################################################################
    with open(work_dir+'/model_params/xgboost_params.txt','w',encoding='utf-8')as f:
        f.write('time:' + local_time + '\n')
        for key in other_params.keys():
            f.write(str(key)+':'+str(other_params[key])+'\n')

    if if_save:
        s = 'Y'
    else:
        s = input('Do you want to save this model?[Y/n]')
    if s == 'Y' or s == 'y' or s == '':
        shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        model_filename = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + 'xgboost_best_model_'+scoring+\
                         '-'+str(period_length)+'-'+str(overlap_ratio)+'.joblib'
        dump(best_model, save_dir + '/' + model_filename)

    return other_params


# 训练XGBoost模型的统一接口，可以选择grid_search调参或直接训练
def train_xgboost(train_data_dir,repo_id,period_length=120,overlap_ratio=0.0,data_type_count=12,scoring='roc_auc',
                   grid_search_control=False,model_params_dir='model_params',prediction_work_dir='.',
                   save_dir='xgboost_models',if_save=True,imp_metric='roc_auc'):
    split_balanced_data_dir = train_data_dir + '/repo_' + str(repo_id) + '/split_balanced_data'
    if grid_search_control:
        gridSearchForXGBoost(split_balanced_data_dir, period_length, overlap_ratio, data_type_count,
                             prediction_work_dir, save_dir, scoring, if_save,imp_metric)
    else:
        model_params_file = model_params_dir + '/xgboost_params.txt'
        params_dict = dict()
        with open(model_params_file, 'r', encoding='utf-8')as f:
            f.readline()
            for line in f.readlines():
                param_name = line.split(':')[0]
                if param_name == 'n_estimators' or param_name == 'min_child_weight' \
                        or param_name == 'max_depth' or param_name=='seed':
                    param_value = int(line.strip('\n').split(':')[1])
                else:
                    param_value = float(line.strip('\n').split(':')[1])
                params_dict[param_name] = param_value
        trainXGBoost(split_balanced_data_dir,period_length,overlap_ratio,data_type_count,params_dict,
                     save_dir=save_dir,if_save=if_save,imp_metric=imp_metric)
