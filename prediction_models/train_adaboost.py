from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from joblib import dump, load
import shutil
import os
from prediction_models.train_svm import getRandomIndex,getModelData
import matplotlib.pyplot as plt
import matplotlib
import time
import pandas as pd
import random


# https://zhuanlan.zhihu.com/p/59121403
def my_adaboost_clf(X_train, Y_train, X_test, Y_test, num_boost_round=300, weak_clf=DecisionTreeClassifier(max_depth=1,random_state=1)):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    for i in range(num_boost_round):
        # Fit a classifier with the specific weights
        weak_clf.fit(X_train, Y_train, sample_weight = w)
        pred_train_i = weak_clf.predict(X_train)
        pred_test_i = weak_clf.predict(X_test)

        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        print("weak_clf_%02d train acc: %.4f"
         % (i + 1, 1 - sum(miss) / n_train))

        # Error,分类误差率，矩阵乘法
        err_m = np.dot(w, miss)
        # Alpha,最终集成使用的的基学习器的权重，误差率小的基学习器拥有较大的权值，误差率大的基学习器拥有较小的权值。
        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
        # New weights
        miss2 = [x if x==1 else -1 for x in miss] # -1 * y_i * G(x_i): 1 / -1
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2])) #对应元素相乘
        w = w / sum(w)

        # Add to prediction，所有弱分类器结果加权求和
        pred_train_i = [1 if x == 1 else -1 for x in pred_train_i]
        pred_test_i = [1 if x == 1 else -1 for x in pred_test_i]
        pred_train = pred_train + np.multiply(alpha_m, pred_train_i)
        pred_test = pred_test + np.multiply(alpha_m, pred_test_i)

    pred_train = [1 if x > 0 else -1 for x in pred_train]
    pred_test = [1 if x > 0 else -1 for x in pred_test]
    return pred_train,pred_test
    # print("My AdaBoost clf train accuracy: %.4f" % (sum(pred_train == Y_train) / n_train))
    # print("My AdaBoost clf test accuracy: %.4f" % (sum(pred_test == Y_test) / n_test))


# 训练自定义的adaboost并返回结果
def trainMyAdaBoost(split_balanced_data_dir,period_length=120,overlap_ratio=0.0,data_type_count=12,num_boost_round=300):
    train_data, test_data, train_label, test_label = getModelData(split_balanced_data_dir, period_length, overlap_ratio,
                                                                  data_type_count)
    train_pred, test_pred = my_adaboost_clf(train_data, train_label, test_data, test_label,
                                            num_boost_round=num_boost_round,
                                            weak_clf=DecisionTreeClassifier(
                                                criterion='gini', max_depth=1,
                                                random_state=1))

    print("训练集：")
    print('Precesion: %.4f' % precision_score(train_label, train_pred))
    print('Recall: %.4f' % recall_score(train_label, train_pred))
    print('F1-score: %.4f' % f1_score(train_label, train_pred, average='binary'))
    print('Accuracy: %.4f' % accuracy_score(train_label, train_pred))
    print('AUC: %.4f' % roc_auc_score(train_label, train_pred))

    print("测试集：")
    print('Precesion: %.4f' % precision_score(test_label, test_pred))
    print('Recall: %.4f' % recall_score(test_label, test_pred))
    print('F1-score: %.4f' % f1_score(test_label, test_pred, average='binary'))
    print('Accuracy: %.4f' % accuracy_score(test_label, test_pred))
    print('AUC: %.4f' % roc_auc_score(test_label, test_pred))


def trainAdaBoost(split_balanced_data_dir,period_length=120,overlap_ratio=0.0,data_type_count=12,
                  dt_max_depth=1,n_estimators=50,learning_rate=1.0,random_state=42,
                  save_dir='adaboost_models',if_save=True):
    train_data, test_data, train_label, test_label = getModelData(split_balanced_data_dir, period_length, overlap_ratio,
                                                                  data_type_count)
    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=dt_max_depth,random_state=1),
                               n_estimators=n_estimators,learning_rate=learning_rate,random_state=random_state)
    model.fit(train_data, train_label)
    y_pred = model.predict(train_data)
    print("训练集：")
    print('Precesion: %.4f' % precision_score(train_label, y_pred))
    print('Recall: %.4f' % recall_score(train_label, y_pred))
    print('F1-score: %.4f' % f1_score(train_label, y_pred, average='binary'))
    print('Accuracy: %.4f' % accuracy_score(train_label, y_pred))
    print('AUC: %.4f' % roc_auc_score(train_label, y_pred))

    y_pred = model.predict(test_data)
    print("测试集：")
    print('Precesion: %.4f' % precision_score(test_label, y_pred))
    print('Recall: %.4f' % recall_score(test_label, y_pred))
    print('F1-score: %.4f' % f1_score(test_label, y_pred, average='binary'))
    print('Accuracy: %.4f' % accuracy_score(test_label, y_pred))
    print('AUC: %.4f' % roc_auc_score(test_label, y_pred))

    if if_save:
        shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        model_filename = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + 'adaboost_model.joblib'
        dump(model, save_dir + '/' + model_filename)


def gridSearchForAdaBoost(split_balanced_data_dir,period_length=120,overlap_ratio=0.0,data_type_count=12,scoring='roc_auc',
                          work_dir='.',save_dir='adaboost_models',if_save=True):
    train_data, test_data, train_label, test_label = getModelData(split_balanced_data_dir, period_length, overlap_ratio,
                                                                  data_type_count)
    model = AdaBoostClassifier(learning_rate=0.1,random_state=42)
    cv_params={'n_estimators':np.linspace(100, 1000, 10, dtype=int)}
    gs=GridSearchCV(model,cv_params,scoring=scoring,verbose=2, refit=True, cv=5, n_jobs=-1)
    gs.fit(train_data,train_label)
    n_estimators = gs.best_params_['n_estimators']
    print('Best n_estimators(approximately):',n_estimators)
    print(scoring + ' score:', gs.best_score_)

    cv_params={'n_estimators':np.linspace(n_estimators-90, n_estimators+90, 19, dtype=int)}
    gs = GridSearchCV(model, cv_params, scoring=scoring, verbose=2, refit=True, cv=5, n_jobs=-1)
    gs.fit(train_data, train_label)
    n_estimators = gs.best_params_['n_estimators']
    print('Best n_estimators(accurately):', n_estimators)
    print(scoring + ' score:', gs.best_score_)

    model = AdaBoostClassifier(n_estimators=n_estimators,learning_rate=0.1,random_state=42)
    max_depth_array = np.linspace(1,10,10,dtype=int)
    dt_list = []
    for max_depth in max_depth_array:
        dt = DecisionTreeClassifier(max_depth=max_depth)
        dt_list.append(dt)
    cv_params={'base_estimator':np.array(dt_list)}
    gs = GridSearchCV(model, cv_params, scoring=scoring, verbose=2, refit=True, cv=5, n_jobs=-1)
    gs.fit(train_data, train_label)
    dt = gs.best_params_['base_estimator']
    print('Best base_estimator(DecisionTreeClassifier):', dt)
    print(scoring + ' score:', gs.best_score_)

    model = AdaBoostClassifier(base_estimator=dt,n_estimators=n_estimators, learning_rate=0.1, random_state=42)
    cv_params = {'learning_rate':np.logspace(-2, 0, 10)}
    gs = GridSearchCV(model, cv_params, scoring=scoring, verbose=2, refit=True, cv=5, n_jobs=-1)
    gs.fit(train_data, train_label)
    learning_rate = gs.best_params_['learning_rate']
    print('Best learning_rate:',learning_rate)
    print(scoring + ' score:', gs.best_score_)

    best_model = gs.best_estimator_
    y_true, y_pred = test_label, best_model.predict(test_data)
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
    with open(work_dir+'/adaboost_result.csv', 'a', encoding='utf-8')as f:
        f.write('time: '+local_time+'\n')
        tmp_index = split_balanced_data_dir.find('repo')
        f.write(split_balanced_data_dir[tmp_index:tmp_index+7]+','+
                str(period_length) + ',' + str(overlap_ratio) +','+str(scoring)+ ',\n')
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
    with open(work_dir+'/model_params/adaboost_params.txt','w',encoding='utf-8')as f:
        f.write('time:' + local_time + '\n')
        f.write('n_estimators:'+str(n_estimators)+'\n')
        f.write('dt_max_depth:'+str(dt).strip(')').split('=')[1]+'\n')
        f.write('learning_rate:'+str(learning_rate)+'\n')

    if if_save:
        s = 'Y'
    else:
        s = input('Do you want to save this model?[Y/n]')
    if s == 'Y' or s == 'y' or s == '':
        shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        model_filename = time.strftime('%Y-%m-%d_%H-%M-%S',
                                       time.localtime()) + 'adaboost_best_model_' + scoring + \
                         '-'+str(period_length)+'-'+str(overlap_ratio)+'.joblib'
        dump(best_model, save_dir + '/' + model_filename)

    return dt,n_estimators,learning_rate


# 训练AdaBoost模型的统一接口，可以选择grid_search调参或直接训练
def train_adaboost(train_data_dir,repo_id,period_length=120,overlap_ratio=0.0,data_type_count=12,scoring='roc_auc',
                   grid_search_control=False,model_params_dir='model_params',prediction_work_dir='.',
                   save_dir='adaboost_models',if_save=True):
    split_balanced_data_dir = train_data_dir + '/repo_' + str(repo_id) + '/split_balanced_data'
    if grid_search_control:
        gridSearchForAdaBoost(split_balanced_data_dir, period_length, overlap_ratio, data_type_count, scoring,
                              prediction_work_dir,save_dir,if_save)
    else:
        model_params_file = model_params_dir + '/adaboost_params.txt'
        params_dict = dict()
        with open(model_params_file, 'r', encoding='utf-8')as f:
            f.readline()
            for line in f.readlines():
                params_dict[line.split(':')[0]]=line.strip('\n').split(':')[1]
        n_estimators = int(params_dict['n_estimators'])
        dt_max_depth = int(params_dict['dt_max_depth'])
        learning_rate = float(params_dict['learning_rate'])
        trainAdaBoost(split_balanced_data_dir,period_length,overlap_ratio,data_type_count,dt_max_depth,n_estimators,
                      learning_rate,save_dir=save_dir,if_save=if_save)