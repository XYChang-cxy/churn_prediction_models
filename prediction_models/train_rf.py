from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from prediction_models.train_svm import getRandomIndex,getModelData
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
            save_dir='rf_models',if_save=True):
    train_data, test_data, train_label, test_label = getModelData(split_balanced_data_dir, period_length, overlap_ratio,
                                                                  data_type_count)
    model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_leaf=min_samples_leaf,
                                   min_samples_split=min_samples_split,max_features=max_features)
    model.fit(train_data, train_label)

    y_pred = model.predict(train_data)
    print("训练集：")
    print('Precesion: %.4f' % precision_score(train_label, y_pred))
    print('Recall: %.4f' % recall_score(train_label, y_pred))
    print('F1-score: %.4f' % f1_score(train_label, y_pred,average='binary'))
    print('Accuracy: %.4f' % accuracy_score(train_label, y_pred))
    print('AUC: %.4f' % roc_auc_score(train_label, y_pred))

    y_pred = model.predict(test_data)
    print("测试集：")
    print('Precesion: %.4f' % precision_score(test_label, y_pred))
    print('Recall: %.4f' % recall_score(test_label, y_pred))
    print('F1-score: %.4f' % f1_score(test_label, y_pred,average='binary'))
    print('Accuracy: %.4f' % accuracy_score(test_label, y_pred))
    print('AUC: %.4f' % roc_auc_score(test_label, y_pred))

    if if_save:
        shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        model_filename = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + 'rf_model.joblib'
        dump(model, save_dir + '/' + model_filename)


# https://www.cnblogs.com/pinard/p/6160412.html
def gridSearchForRF(split_balanced_data_dir,period_length=120,overlap_ratio=0.0,data_type_count=12,
                    scoring='roc_auc',work_dir='.',save_dir='rf_models',if_save=True):
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
    y_true,y_pred = test_label,best_model.predict(test_data)
    print('best model classification report:\n')
    print(classification_report(y_true, y_pred))

    test_pred = y_pred
    print("\n测试集结果：")
    print("accuracy score:\t", accuracy_score(test_label, test_pred))
    print("auroc:\t", roc_auc_score(test_label, test_pred))
    print("precision:\t", precision_score(test_label, test_pred))
    print("recall:\t", recall_score(test_label, test_pred))
    print("f1_score:\t", f1_score(test_label, test_pred, average='binary'))

    train_pred = best_model.predict(train_data)

    local_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())

    ################################################################################3
    with open(work_dir+'/rf_result.csv', 'a', encoding='utf-8')as f:
        f.write('time: ' + local_time + '\n')
        tmp_index = split_balanced_data_dir.find('repo')
        f.write(split_balanced_data_dir[tmp_index:tmp_index + 7] + ',' +
                str(period_length) + ',' + str(overlap_ratio) + ',' + str(scoring) + ',\n')
        f.write('train accuracy,' + str(accuracy_score(train_label, train_pred)) + ',\n')
        f.write('train precision,' + str(precision_score(train_label, train_pred)) + ',\n')
        f.write('train recall,' + str(recall_score(train_label, train_pred)) + ',\n')
        f.write('train f1_score,' + str(f1_score(train_label, train_pred, average='binary')) + ',\n')
        f.write('train auroc,' + str(roc_auc_score(train_label, train_pred)) + ',\n')

        f.write('test accuracy,' + str(accuracy_score(test_label, test_pred)) + ',\n')
        f.write('test precision,' + str(precision_score(test_label, test_pred)) + ',\n')
        f.write('test recall,' + str(recall_score(test_label, test_pred)) + ',\n')
        f.write('test f1_score,' + str(f1_score(test_label, test_pred, average='binary')) + ',\n')
        f.write('test auroc,' + str(roc_auc_score(test_label, test_pred)) + ',\n')
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
             save_dir='rf_models',if_save=True):
    split_balanced_data_dir = train_data_dir + '/repo_' + str(repo_id) + '/split_balanced_data'
    if grid_search_control:
        gridSearchForRF(split_balanced_data_dir, period_length, overlap_ratio, data_type_count, scoring,
                        prediction_work_dir, save_dir, if_save)
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
                min_samples_leaf,min_samples_split,max_features,save_dir,if_save)