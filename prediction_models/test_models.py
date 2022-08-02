from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,classification_report
from prediction_models.train_svm import *
from prediction_models.train_rf import *
from prediction_models.train_xgboost import *
from prediction_models.train_adaboost import *
from data_preprocess.preprocess import get_existed_prediction_data
from prediction_models.result_analysis import get_proba_error,get_proba_metric
import time
from collections import Counter
from joblib import dump, load
import os


# precision偏小，表明loyaler被判定为churner的较多
# recall偏小，表明churner被判定为loyaler的较多

if __name__ == '__main__':
    # id = 25
    id_list = [20]
    for id in id_list:
        split_balanced_data_dir = 'F:/MOOSE_cxy/churn_prediction/data_preprocess/data/repo_'\
                                  +str(id)+'/part_all/split_balanced_data'
        period_length = 120
        overlap_ratio = 0.0
        data_type_count = 9  # Gitee
        scoring = 'precision'

        # getModelData(split_balanced_data_dir,period_length,overlap_ratio)

        # trainSVM(split_balanced_data_dir, period_length, overlap_ratio, data_type_count)
        # trainRF(split_balanced_data_dir,period_length,overlap_ratio,data_type_count)
        # trainXGBoost(split_balanced_data_dir,period_length,overlap_ratio,data_type_count)
        # trainMyAdaBoost(split_balanced_data_dir,period_length,overlap_ratio,data_type_count,num_boost_round=100)
        # trainAdaBoost(split_balanced_data_dir,period_length,overlap_ratio,data_type_count)

        # scoring: accuracy、precision、recall、f1、f1_micro、f1_macro、roc_auc
        # gridSearchForSVM(split_balanced_data_dir, period_length, overlap_ratio, data_type_count,scoring='precision')
        # gridSearchForRF(split_balanced_data_dir, period_length, overlap_ratio, data_type_count,scoring=scoring)
        # gridSearchForAdaBoost(split_balanced_data_dir, period_length, overlap_ratio, data_type_count,scoring=scoring)

        # gridSearchForSVM(split_balanced_data_dir, period_length, overlap_ratio, data_type_count, scoring='roc_auc')
        # gridSearchForRF(split_balanced_data_dir, period_length, overlap_ratio, data_type_count,scoring='roc_auc')
        # gridSearchForXGBoost(split_balanced_data_dir, period_length, overlap_ratio, data_type_count,scoring=scoring)
        # gridSearchForAdaBoost(split_balanced_data_dir, period_length, overlap_ratio, data_type_count,scoring='roc_auc')

        # split_balanced_data_dir = r'F:\MOOSE_cxy\developer_churn_prediction\churn_prediction\data_preprocess\data\repo_2\part_2\split_balanced_data'

        split_balanced_data_dir = '../data_preprocess/train_data/repo_8649239/split_balanced_data'
        period_length_list = [120]#120,30
        overlap_ratio_list = [0.0]
        scoring_list = ['roc_auc']#,'accuracy','precision'
        for scoring in scoring_list:
            for period in period_length_list:
                for overlap in overlap_ratio_list:
                    pass
                    gridSearchForSVM(split_balanced_data_dir, period, overlap, data_type_count,scoring=scoring)
                    # trainSVM(split_balanced_data_dir,period,overlap,data_type_count)
                    gridSearchForRF(split_balanced_data_dir, period, overlap, data_type_count, scoring=scoring)
                    time.sleep(10)
                    gridSearchForAdaBoost(split_balanced_data_dir, period, overlap, data_type_count,
                                          scoring=scoring)
                    time.sleep(10)
                    gridSearchForXGBoost(split_balanced_data_dir, period, overlap, data_type_count,
                                         scoring=scoring)
                    time.sleep(20)

    # split_balanced_data_dir = '../data_preprocess/train_data/repo_8649239/split_balanced_data'
    # train_data, test_data, train_label, test_label = getModelData(split_balanced_data_dir, period_length, overlap_ratio,
    #                                                               data_type_count)
    # model_path = '../prediction_models/rf_models/2022-06-17_10-49-37rf_best_model_roc_auc-120-0.0.joblib'
    # model_type = model_path.split('/')[-1][19:].split('_')[0]
    # print(model_type)
    # if model_type == 'xgboost':
    #     for i in range(train_label.shape[0]):
    #         if train_label[i] == -1:
    #             train_label[i] = 0
    #     for i in range(test_label.shape[0]):
    #         if test_label[i] == -1:
    #             test_label[i] = 0
    #
    # model = load(model_path)
    # train_pred = model.predict(train_data)
    # test_pred = model.predict(test_data)
    #
    # for i in range(len(test_label)):
    #     if test_label[i] != test_pred[i]:
    #         print(test_label[i], test_pred[i])
    #
    # print(Counter(test_label), Counter(test_pred))
    #
    # print("训练集结果：")
    # print("accuracy score:\t", accuracy_score(train_label, train_pred))
    # print("auroc:\t", roc_auc_score(train_label, train_pred))
    # print("precision:\t", precision_score(train_label, train_pred))
    # print("recall:\t", recall_score(train_label, train_pred))
    # print("f1_score:\t", f1_score(train_label, train_pred, average='binary'))
    #
    # print("\n测试集结果：")
    # print("accuracy score:\t", accuracy_score(test_label, test_pred))
    # print("auroc:\t", roc_auc_score(test_label, test_pred))
    # print("precision:\t", precision_score(test_label, test_pred))
    # print("recall:\t", recall_score(test_label, test_pred))
    # print("f1_score:\t", f1_score(test_label, test_pred, average='binary'))

    '''model_type = 'xgboost'  # svm,rf,xgboost,adaboost
    filenames = os.listdir(model_type+'_models')
    # scoring='f1'
    # period_length = 120
    # overlap_ratio = 0.5
    filename = ''
    for name in filenames:
        if name.split('-')[-2]==str(period_length) and name.split('-')[-1]==str(overlap_ratio)+'.joblib'\
                and name.split('-')[-3].split('_')[-1]==scoring.split('_')[-1]:
            filename = model_type+'_models/'+name
            break
    model = load(filename)
    train_data, test_data, train_label, test_label = getModelData(split_balanced_data_dir, period_length, overlap_ratio,
                                                                  data_type_count)
    if model_type == 'xgboost':
        for i in range(train_label.shape[0]):
            if train_label[i] == -1:
                train_label[i] = 0
        for i in range(test_label.shape[0]):
            if test_label[i] == -1:
                test_label[i] = 0

    train_pred = model.predict(train_data)
    test_pred = model.predict(test_data)

    for i in range(len(test_label)):
        if test_label[i]!=test_pred[i]:
            print(test_label[i],test_pred[i])

    print(Counter(test_label),Counter(test_pred))

    print("训练集结果：")
    print("accuracy score:\t", accuracy_score(train_label, train_pred))
    print("auroc:\t", roc_auc_score(train_label, train_pred))
    print("precision:\t", precision_score(train_label, train_pred))
    print("recall:\t", recall_score(train_label, train_pred))
    print("f1_score:\t", f1_score(train_label, train_pred, average='binary'))


    print("\n测试集结果：")
    print("accuracy score:\t", accuracy_score(test_label, test_pred))
    print("auroc:\t", roc_auc_score(test_label, test_pred))
    print("precision:\t", precision_score(test_label, test_pred))
    print("recall:\t", recall_score(test_label, test_pred))
    print("f1_score:\t", f1_score(test_label, test_pred, average='binary'))'''

    # 生成预测数据集，并返回
    # user_id_list,input_data = prediction_data_preprocess(repo_id,train_data_dir,prediction_data_dir,
    #                                                      period_length,churn_limit_weeks,time_threshold_days)
    # print(user_id_list)
    # print(input_data)

    prediction_file = '../data_preprocess/prediction_data/repo_8649239/normalized_data'

    # 根据生成的预测数据集文件，直接返回数据
    user_id_list, input_data = get_existed_prediction_data(prediction_file)
    print(len(user_id_list))
    print(input_data.shape)

    # 加载训练好的模型，对input_data进行预测
    # model_path = './prediction_models/xgboost_models/2022-06-03_15-16-04xgboost_best_model_roc_auc-120-0.0.joblib'
    '''model_path = '../prediction_models/xgboost_models/2022-07-03_11-07-03xgboost_best_model_roc_auc-120-0.0.joblib'
    # model_path = './prediction_models/rf_models/2022-06-03_15-11-48rf_best_model_roc_auc-120-0.0.joblib'
    # model_path = './prediction_models/adaboost_models/2022-06-03_15-15-13adaboost_best_model_roc_auc-120-0.0.joblib'
    # model_path = '../prediction_models/svm_models/2022-07-03_10-37-00svm_best_model_roc_auc-120-0.0.joblib'

    model = load(model_path)
    y_pred = model.predict(input_data)
    y_pred_proba = model.predict_proba(input_data)
    for i in range(len(y_pred)):
        if y_pred_proba[i][1] > 0.5:
            print(y_pred[i], 1, y_pred_proba[i])
        else:
            print(y_pred[i], -1, y_pred_proba[i])
    y_pred_proba = np.array([x[1] for x in y_pred_proba])
    # get_proba_error(y_pred,y_pred_proba,['MSE','RMSE','MAE','MAPE','SMAPE','R2'])
    print(get_proba_metric(y_pred,y_pred_proba))'''

