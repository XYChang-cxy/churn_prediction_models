from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score
import numpy as np
from prettytable import PrettyTable

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

# 预测结果
# 参考：https://blog.csdn.net/weixin_45901519/article/details/107006208
# y_true: 真实值，正类用1表示，负类用0或-1表示
# y_pred_proba: 概率结果列表，概率值越接近于1则是正类，越接近于0则是负类
# error_type_list: 误差类型，可选项包括:
#                  MSE--均方误差
#                  RMSE--均方根误差
#                  MAE--平均绝对值误差
#                  MAPE--平均绝对值百分比误差
#                  SMAPE--对称平均绝对值百分比误差
#                  R2--R方
def get_proba_error(y_true,y_pred_proba, error_type_list=None,verbose=True):
    if error_type_list is None:
        error_type_list = ['MSE']
    y_true = np.array([max(x,0) for x in y_true])
    error_results = dict()
    for error_type in error_type_list:
        if error_type == 'MSE':
            error_results[error_type]=mean_squared_error(y_true,y_pred_proba)
        elif error_type == 'RMSE':
            error_results[error_type]=np.sqrt(mean_squared_error(y_true,y_pred_proba))
        elif error_type == 'MAE':
            error_results[error_type]=mean_absolute_error(y_true,y_pred_proba)
        elif error_type == 'MAPE':
            error_results[error_type]=mean_absolute_percentage_error(y_true,y_pred_proba)
        elif error_type == 'SMAPE':
            error_results[error_type]=smape(y_true,y_pred_proba)
        elif error_type == 'R2':
            error_results[error_type]=r2_score(y_true,y_pred_proba)
        else:
            print('Error type error!',error_type+' does not exist!')
            return
    if verbose:
        tab = PrettyTable(['Error Name','Value'])
        for error_type in error_type_list:
            tab.add_row([error_type,str(error_results[error_type])])
        print(tab)
    return error_results


# metric_name:可选参数包含：accuracy,roc_auc,precision,recall,f1_score
def get_metric_value(metric_name,y_true,y_pred):
    if metric_name=='accuracy':
        return accuracy_score(y_true,y_pred)
    elif metric_name == 'roc_auc':
        return roc_auc_score(y_true,y_pred)
    elif metric_name =='precision':
        return precision_score(y_true,y_pred)
    elif metric_name == 'recall':
        return recall_score(y_true,y_pred)
    elif metric_name == 'f1_score':
        return f1_score(y_true,y_pred)
    else:
        return None


# 根据阈值，将概率值转变为二分值，并计算accuracy等值
# threshold: 0~1之间的小数，表示划分正负类的阈值；若为None，则自动探索最优阈值
# imp_metric:根据哪个指标来选取最优的阈值，可选参数包含：accuracy,roc_auc,precision,recall,f1_score
def get_proba_metric(y_true,y_pred_proba,threshold=None,imp_metric='accuracy',verbose=True):
    y_true = np.array([max(x,0) for x in y_true])
    if threshold is not None:
        y_pred = np.array([1 if x >= threshold else 0 for x in y_pred_proba])
        if verbose:
            tmp_list = []
            tmp_list.append(accuracy_score(y_true, y_pred))
            tmp_list.append(roc_auc_score(y_true, y_pred))
            tmp_list.append(precision_score(y_true, y_pred))
            tmp_list.append(recall_score(y_true, y_pred))
            tmp_list.append(f1_score(y_true, y_pred,average='binary'))
            tab = PrettyTable(['Metric Name', 'Value'])
            tab.add_row(['accuracy score', str(tmp_list[0])])
            tab.add_row(['roc_auc', str(tmp_list[1])])
            tab.add_row(['precision', str(tmp_list[2])])
            tab.add_row(['recall', str(tmp_list[3])])
            tab.add_row(['f1_socre', str(tmp_list[4])])
            print(tab)
            return threshold,tmp_list
    else:
        threshold_results = dict()
        threshold_array = np.linspace(0.1, 0.9, 9, dtype=float)
        max_metric = 0
        max_metric_thres = 0
        for threshold in threshold_array:
            y_pred = np.array([1 if x >= threshold else 0 for x in y_pred_proba])
            tmp_list = []
            val = get_metric_value(imp_metric,y_true,y_pred)
            if val > max_metric:
                max_metric = val
                max_metric_thres = threshold
            tmp_list.append(accuracy_score(y_true,y_pred))
            tmp_list.append(roc_auc_score(y_true,y_pred))
            tmp_list.append(precision_score(y_true,y_pred))
            tmp_list.append(recall_score(y_true,y_pred))
            tmp_list.append(f1_score(y_true,y_pred,average='binary'))
            threshold_results[threshold]=tmp_list.copy()
        threshold_array_2 = np.linspace(max_metric_thres - 0.08, max_metric_thres + 0.08, 9, dtype=float)
        for threshold in threshold_array_2:
            y_pred = np.array([1 if x >= threshold else 0 for x in y_pred_proba])
            tmp_list = []
            val = get_metric_value(imp_metric, y_true, y_pred)
            if val > max_metric:
                max_metric = val
                max_metric_thres = threshold
            tmp_list.append(accuracy_score(y_true,y_pred))
            tmp_list.append(roc_auc_score(y_true,y_pred))
            tmp_list.append(precision_score(y_true,y_pred))
            tmp_list.append(recall_score(y_true,y_pred))
            tmp_list.append(f1_score(y_true,y_pred,average='binary'))
            threshold_results[threshold]=tmp_list.copy()
        if verbose:
            tab = PrettyTable(['threshold','accuracy score','roc_auc','precision','recall','f1_score'])
            for thres in threshold_array:
                if thres < threshold_array_2[0] or thres > threshold_array_2[-1]:
                    new_list = [float('{0:.2f}'.format(thres))]
                    new_list.extend(threshold_results[thres])
                    tab.add_row(new_list)
                else:
                    for thres2 in threshold_array_2:
                        new_list = [float('{0:.2f}'.format(thres2))]
                        new_list.extend(threshold_results[thres2])
                        tab.add_row(new_list)
            print(tab)
        return max_metric_thres,threshold_results[max_metric_thres]




