from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from joblib import dump, load
import matplotlib.pyplot as plt
import matplotlib
import time
import pandas as pd
import random
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
    elif period_length == 30:
        col_count = 6*data_type_count
    else:
        print('period length error!')
        return

    data_filename_1 = split_balanced_data_dir + '\\balanced_data_train-' + str(period_length) + '-' + str(
        overlap_ratio) + '.csv'
    data_filename_2 = split_balanced_data_dir + '\\balanced_data_test-' + str(period_length) + '-' + str(
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
def trainSVM(split_balanced_data_dir,period_length=120,overlap_ratio=0.0,data_type_count=12,
             kernel='rbf',C=1,gamma='auto',degree=3,
             save_dir='svm_models'):
    train_data,test_data,train_label,test_label = getModelData(split_balanced_data_dir,period_length,overlap_ratio,
                                                               data_type_count)
    if kernel == 'rbf' or kernel == 'RBF':
        classifier = svm.SVC(kernel='rbf', C=C, gamma=gamma)
    elif kernel == 'poly':
        classifier = svm.SVC(kernel='poly', C=C, degree=degree)
    else:
        classifier = svm.SVC(kernel='linear', C=C)

    classifier.fit(train_data, train_label)

    train_pred = classifier.predict(train_data)
    test_pred = classifier.predict(test_data)

    print("训练集结果：")
    print("accuracy score:\t", accuracy_score(train_label, train_pred))
    print("auroc:\t",roc_auc_score(train_label,train_pred))
    print("precision:\t", precision_score(train_label, train_pred))
    print("recall:\t", recall_score(train_label, train_pred))
    print("f1_score:\t", f1_score(train_label, train_pred,average='binary'))

    '''count = 0
    for i in range(len(train_pred)):
        if train_label[i] * train_pred[i]<0:
            print(i,train_label[i],train_pred[i])
            count += 1
    print(count)'''

    print("\n测试集结果：")
    print("accuracy score:\t", accuracy_score(test_label, test_pred))
    print("auroc:\t", roc_auc_score(test_label,test_pred))
    print("precision:\t", precision_score(test_label, test_pred))
    print("recall:\t", recall_score(test_label, test_pred))
    print("f1_score:\t", f1_score(test_label, test_pred, average='binary'))

    '''count = 0
    for i in range(len(test_pred)):
        if test_label[i] * test_pred[i] < 0:
            print(i, test_label[i], test_pred[i])
            count += 1
    print(count)'''

    s = input('Do you want to save this model?[Y/n]')
    if s == 'Y' or s == 'y' or s == '':
        model_filename = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + 'svm_model.joblib'
        dump(classifier, save_dir + '\\' + model_filename)


# GridSearch调参
# https://blog.csdn.net/aliceyangxi1987/article/details/73769950
def gridSearchForSVM(split_balanced_data_dir,period_length=120,overlap_ratio=0.0,data_type_count=12,
                     scoring='roc_auc',save_dir='svm_models',if_save=True):
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
        clf = GridSearchCV(svm.SVC(), tuned_parameters, return_train_score=True,
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
        y_true,y_pred = test_label,best_model.predict(test_data)
        print('best model classification report:\n')
        print(classification_report(y_true,y_pred))

        test_pred=y_pred
        print("\n测试集结果：")
        print("accuracy score:\t", accuracy_score(test_label, test_pred))
        print("auroc:\t", roc_auc_score(test_label, test_pred))
        print("precision:\t", precision_score(test_label, test_pred))
        print("recall:\t", recall_score(test_label, test_pred))
        print("f1_score:\t", f1_score(test_label, test_pred, average='binary'))

        train_pred = best_model.predict(train_data)

        ################################################################################3
        with open('svm_result.csv', 'a', encoding='utf-8')as f:
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
            model_filename = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + 'svm_best_model_'+score+\
                             '-'+str(period_length)+'-'+str(overlap_ratio)+'.joblib'
            dump(best_model, save_dir + '\\' + model_filename)

    return best_params_dict


if __name__ == '__main__':
    split_balanced_data_dir = r'F:\MOOSE_cxy\developer_churn_prediction\churn_prediction\data_preprocess\data\repo_2\part_all\balanced_data'
    period_length = 120
    overlap_ratio = 0.0
    data_type_count = 12
    # gridSearchForSVM(split_balanced_data_dir,period_length,overlap_ratio,data_type_count)
    trainSVM(split_balanced_data_dir,period_length,overlap_ratio,data_type_count)

