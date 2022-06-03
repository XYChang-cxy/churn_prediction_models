import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from prediction_models.train_svm import getRandomIndex,getModelData
from joblib import dump, load
import matplotlib.pyplot as plt
import matplotlib
import xgboost as xgb
import time
import pandas as pd
import random


def trainXGBoost(split_balanced_data_dir,period_length=120,overlap_ratio=0.0,data_type_count=12,
                 n_estimators=500,max_depth=6,min_child_weight=1,subsample=1,colsample_bytree=1,
                 gamma=0,reg_lambda=1,reg_alpha=0,eta=0.3,
                 save_dir='xgboost_models'):
    train_data, test_data, train_label, test_label = getModelData(split_balanced_data_dir, period_length, overlap_ratio,
                                                                  data_type_count)
    for i in range(train_label.shape[0]):
        if train_label[i]==-1:
            train_label[i]=0
    for i in range(test_label.shape[0]):
        if test_label[i]==-1:
            test_label[i]=0

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

    s = input('Do you want to save this model?[Y/n]')
    if s == 'Y' or s == 'y' or s == '':
        model_filename = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + 'xgboost_model.joblib'
        dump(model, save_dir + '\\' + model_filename)


def gridSearchForXGBoost(split_balanced_data_dir,period_length=120,overlap_ratio=0.0,data_type_count=12,
                         save_dir='xgboost_models',scoring='roc_auc',if_save=True):
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
    y_true,y_pred=test_label,best_model.predict(test_data)
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

    ################################################################################3
    with open('xgboost_result.csv','a',encoding='utf-8')as f:
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

    if if_save:
        s = 'Y'
    else:
        s = input('Do you want to save this model?[Y/n]')
    if s == 'Y' or s == 'y' or s == '':
        model_filename = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + 'xgboost_best_model_'+scoring+\
                         '-'+str(period_length)+'-'+str(overlap_ratio)+'.joblib'
        dump(best_model, save_dir + '\\' + model_filename)

    return other_params
