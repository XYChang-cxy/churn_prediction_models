# 该文件用于获取某个社区一定时间范围内所有开发者活跃时间段，按周进行统计，时间精确到周
from data_preprocess.database_connect import *
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 获取社区一段时间内的开发者id列表，不考虑fork和star的用户
# repo_id:仓库id
# startDay:筛选时间段的起始日期（包含）
# endDay:筛选时间段的终止日期（不含）
# exceptList:排除的用户id
def getRepoUserList(repo_id,startDay,endDay,exceptList=None):
    if exceptList is None:
        exceptList = []
    userList = []

    results = getRepoDataFromTable(['user_id'],'repo_issue',repo_id,startDay,endDay,'create_time',is_distinct=True)
    for result in results:
        if result[0] not in userList and result[0] not in exceptList:
            userList.append(result[0])

    results = getRepoDataFromTable(['user_id'],'repo_issue_comment',repo_id,startDay,endDay,'create_time',is_distinct=True)
    for result in results:
        if result[0] not in userList and result[0] not in exceptList:
            userList.append(result[0])

    results = getRepoDataFromTable(['user_id'],'repo_pull',repo_id,startDay,endDay,'create_time',is_distinct=True)
    for result in results:
        if result[0] not in userList and result[0] not in exceptList:
            userList.append(result[0])

    results = getRepoDataFromTable(['user_id'],'repo_review_comment',repo_id,startDay,endDay,'create_time',is_distinct=True)
    for result in results:
        if result[0] not in userList and result[0] not in exceptList:
            userList.append(result[0])

    '''results = getRepoDataFromTable(['user_id'],'repo_commit',repo_id,startDay,endDay,is_distinct=True)
    for result in results:
        if result[0] not in userList and result[0] not in exceptList:
            userList.append(result[0])

    results = getRepoDataFromTable(['user_id'],'repo_commit_comment',repo_id,startDay,endDay,is_distinct=True)
    for result in results:
        if result[0] not in userList and result[0] not in exceptList:
            userList.append(result[0])'''
    return userList


# 获取社区一段时间内所有开发者的活跃时间段并存储
# startDay，endDay: 统计的开始和结束时间
# save_dir:存储文件夹
# time_threshold_percentile: 划分开发者的阈值（时间满足大于80%）
# 返回：活跃时间长度的第80百分位数（作为划分开发者的阈值之一）
def saveUserActivePeriod(repo_id,startDay,endDay,save_dir,churn_limit_weeks,time_threshold_percentile=80):
    user_inactive_weeks = dict()  # 每周统计一次
    user_retain_weeks = dict()  # 从用户加入/回访社区开始到当前的周数,每周统计一次
    all_churn_user_list = []  # 累计流失用户
    retain_user_list = []  # 留存用户列表

    timedelta = datetime.datetime.strptime(endDay, fmt_day) - datetime.datetime.strptime(startDay, fmt_day)
    period_weeks = int(timedelta.days / 7)  # 时间段总周数

    filename = save_dir+'/'+str(repo_id)+'_user_active_period.csv'
    with open(filename,'w',encoding='utf-8')as f:
        f.write('user id,start day,end day,days,is_churner,\n')
    f.close()

    time_list = []

    for i in range(period_weeks):
        print(i,'/',period_weeks)
        churn_user_list = []  # 每个统计间隔内流失用户列表
        start_day = (datetime.datetime.strptime(startDay,fmt_day)+datetime.timedelta(days=i*7)).strftime(fmt_day)
        end_day = (datetime.datetime.strptime(start_day,fmt_day)+datetime.timedelta(days=7)).strftime(fmt_day)
        week_user_list = getRepoUserList(repo_id,start_day,end_day)  # 获取本周有活动的开发者
        for user_id in user_inactive_weeks.keys():
            if user_id in week_user_list and user_inactive_weeks[user_id] > 0:  # 上周无活动，本周有活动
                user_inactive_weeks[user_id] = 0
                if user_id in churn_user_list:  # 流失用户回归
                    churn_user_list.remove(user_id)
                if user_id in all_churn_user_list:  # 流失用户回归
                    all_churn_user_list.remove(user_id)
            elif user_id not in week_user_list:  # 本周没有活动
                user_inactive_weeks[user_id] += 1
                if user_inactive_weeks[user_id] >= churn_limit_weeks and user_id not in all_churn_user_list:
                    if user_inactive_weeks[user_id] > churn_limit_weeks:  #########################################
                        print(repo_id, i, user_id, churn_limit_weeks, user_inactive_weeks[user_id])
                    churn_user_list.append(user_id)
                    all_churn_user_list.append(user_id)
                    if user_id in retain_user_list:
                        retain_user_list.remove(user_id)
        for user_id in week_user_list:
            if user_id not in retain_user_list:  # 回访用户/新用户
                retain_user_list.append(user_id)
                user_retain_weeks[user_id] = 0  # 后续统一加1
            if user_id not in user_inactive_weeks.keys():
                user_inactive_weeks[user_id] = 0
        for user_id in retain_user_list:
            user_retain_weeks[user_id] += 1  # 所有留存用户的贡献周数加一

        with open(filename, 'a', encoding='utf-8')as f:
            for user_id in churn_user_list:
                end = (datetime.datetime.strptime(end_day,fmt_day)-
                       datetime.timedelta(days=user_inactive_weeks[user_id]*7)).strftime(fmt_day)
                start = (datetime.datetime.strptime(end_day,fmt_day)-
                         datetime.timedelta(days=(user_retain_weeks[user_id]+1)*7)).strftime(fmt_day)
                days = (datetime.datetime.strptime(end,fmt_day)-datetime.datetime.strptime(start,fmt_day)).days
                time_list.append(days)
                f.write(str(user_id)+','+start+','+end+','+str(days)+',1,\n')
        f.close()

    print(retain_user_list)

    with open(filename,'a',encoding='utf-8')as f:
        for user_id in retain_user_list:
            end = (datetime.datetime.strptime(startDay,fmt_day)+
                   datetime.timedelta(days=period_weeks*7)-
                   datetime.timedelta(days=user_inactive_weeks[user_id]*7)).strftime(fmt_day)
            start = (datetime.datetime.strptime(startDay,fmt_day)+
                     datetime.timedelta(days=period_weeks*7)-
                     datetime.timedelta(days=user_retain_weeks[user_id]*7)).strftime(fmt_day)
            days = (datetime.datetime.strptime(end, fmt_day) - datetime.datetime.strptime(start, fmt_day)).days
            time_list.append(days)
            f.write(str(user_id) + ',' + start + ',' + end + ',' + str(days) + ',0,\n')
    f.close()

    data_array = np.array(time_list)
    time_threshold = np.percentile(data_array,time_threshold_percentile)
    print(time_threshold)
    with open(filename, 'a', encoding='utf-8')as f:
        f.write('time threshold('+str(time_threshold_percentile)+'%),'+str(time_threshold)+',\n')
    return time_threshold


# 获取用于模型训练的用户和对应活跃时间
# user_type: 用户类型，'churner','loyaler'
# user_active_period_file: 存储所有用户活跃时间段的文件
# churn_limit_weeks: 流失期限（单位为周）
# period_length: 获取数据的时间段长度，该长度固定，单位为天，120/90/60/30
# overlap_ratio: 当user_type='loyaler'时有效，表示获取历史数据时同一开发者不同数据区间的重合度
# time_threshold:划分开发者的时间阈值，可以指定，如果是-1则使用user_active_period_file中的阈值(80百分位数)
def getModelUserPeriod(repo_id,user_type,save_dir,user_active_period_file,churn_limit_weeks,period_length=120,overlap_ratio=0.0,
                       time_threshold=-1):
    if user_type!='churner' and user_type !='loyaler':
        print('User type error!')
        return
    if time_threshold == -1:
        time_threshold = float(pd.read_csv(user_active_period_file,usecols=[1]).iloc[-1].iat[0])
    # print(time_threshold)
    user_periods=[]
    with open(user_active_period_file,'r',encoding='utf-8')as f:
        f.readline()
        for line in f.readlines():
            items = line.strip(',\n').split(',')
            if len(items) < 5 or int(items[3])<time_threshold or items[0]=='0':
                continue
            if user_type == 'churner' and items[-1]=='0':
                continue
            if user_type == 'churner':
                new_list = []
                new_list.append(items[0])
                end_day = items[2]
                start_day = (datetime.datetime.strptime(end_day,fmt_day)-
                             datetime.timedelta(days=period_length)).strftime(fmt_day)
                new_list.append(start_day)
                new_list.append(end_day)
                user_periods.append(new_list.copy())
            else:
                start = items[1]
                end = items[2]
                churn_limit = churn_limit_weeks*7  # 流失期限单位转为天
                end_day = (datetime.datetime.strptime(end,fmt_day)-
                           datetime.timedelta(days=churn_limit)).strftime(fmt_day)  # 从后往前截取时间段，起始截取点
                time_delta = (datetime.datetime.strptime(end_day,fmt_day)-
                              datetime.datetime.strptime(start,fmt_day)).days
                while time_delta>= time_threshold:  # 保证每组数据的有效部分长度均大等于time_threshold
                    start_day = (datetime.datetime.strptime(end_day, fmt_day) -
                                 datetime.timedelta(days=period_length)).strftime(fmt_day)
                    user_periods.append([items[0],start_day,end_day])
                    end_day = (datetime.datetime.strptime(end_day,fmt_day)-
                               datetime.timedelta(days=int(period_length*(1-overlap_ratio)))).strftime(fmt_day)
                    time_delta = (datetime.datetime.strptime(end_day, fmt_day) -
                                  datetime.datetime.strptime(start, fmt_day)).days

    f.close()
    if user_type == 'loyaler':
        filename = save_dir + '/' + 'repo_' + str(user_type) + 's_period-' + str(period_length) + \
                   '-'+str(overlap_ratio)+'.csv'
    else:
        filename = save_dir + '/' + 'repo_' + str(user_type) + 's_period-' + str(period_length) + '.csv'
    with open(filename,'w',encoding='utf-8')as f:
        f.write('user_id,start_day,end_day,\n')
        for user_period in user_periods:
            line = user_period[0]+','+str(user_period[1])+','+str(user_period[2])+','
            print((datetime.datetime.strptime(user_period[2],fmt_day)-
                   datetime.datetime.strptime(user_period[1],fmt_day)).days)
            f.write(line+'\n')
    f.close()


# 获取用于模型实际预测的用户和对应活跃时间
# repo_id:仓库id
# user_active_period_file: 存储所有用户活跃时间段的文件
# period_length: 获取数据的时间段长度，该长度固定，单位为天，120/90/60/30
# time_threshold:划分开发者的时间阈值，可以指定，如果是-1则使用user_active_period_file中的阈值(80百分位数)
def getPredictionUserPeriod(repo_id,save_dir,user_active_period_file,period_length=120,time_threshold=-1):
    if time_threshold == -1:
        time_threshold = float(pd.read_csv(user_active_period_file,usecols=[1]).iloc[-1].iat[0])
    # print(time_threshold)
    user_periods=[]
    with open(user_active_period_file,'r',encoding='utf-8')as f:
        f.readline()
        for line in f.readlines():
            items = line.strip(',\n').split(',')
            if len(items) < 5 or int(items[3])<time_threshold or items[0]=='0':
                continue
            if items[-1]=='1': # 流失的开发者不再考虑
                continue
            new_list = []
            new_list.append(items[0])
            end_day = items[2]
            start_day = (datetime.datetime.strptime(end_day, fmt_day) -
                         datetime.timedelta(days=period_length)).strftime(fmt_day)
            new_list.append(start_day)
            new_list.append(end_day)
            user_periods.append(new_list.copy())

    f.close()
    filename = save_dir + '/' + 'repo_users_period-' + str(period_length) + '.csv'
    with open(filename,'w',encoding='utf-8')as f:
        f.write('user_id,start_day,end_day,\n')
        for user_period in user_periods:
            line = user_period[0]+','+str(user_period[1])+','+str(user_period[2])+','
            print((datetime.datetime.strptime(user_period[2],fmt_day)-
                   datetime.datetime.strptime(user_period[1],fmt_day)).days)
            f.write(line+'\n')
    f.close()


# # 以下仅仅是测试
# if __name__ == '__main__':
#     # saveUserActivePeriod(2,'2012-12-21','2022-01-01',r'C:\Users\cxy\Desktop\test')
#     # saveUserActivePeriod(2, '2012-12-21', '2018-02-23', r'C:\Users\cxy\Desktop\test\part1')
#     # saveUserActivePeriod(2, '2018-02-23', '2022-01-01', r'C:\Users\cxy\Desktop\test\part2')
#
#     # getModelUserPeriod(2,'loyaler',r'C:\Users\cxy\Desktop\test',r'C:\Users\cxy\Desktop\test\2_user_active_period.csv',
#     #                    period_length=30,overlap_ratio=0.5)
#     # getModelUserPeriod(2, 'loyaler', r'C:\Users\cxy\Desktop\test\part1',
#     #                    r'C:\Users\cxy\Desktop\test\part1\2_user_active_period.csv',
#     #                    period_length=30, overlap_ratio=0.5)
#     # getModelUserPeriod(2, 'loyaler', r'C:\Users\cxy\Desktop\test\part2',
#     #                    r'C:\Users\cxy\Desktop\test\part2\2_user_active_period.csv',
#     #                    period_length=30, overlap_ratio=0.5)
#
#     saveUserActivePeriod(2,'2012-12-21','2022-01-01',
#                          r'F:\MOOSE_cxy\developer_churn_prediction\churn_prediction\data_preprocess\data\repo_2\part_all')
#     saveUserActivePeriod(2, '2012-12-21', '2018-02-23',
#                          r'F:\MOOSE_cxy\developer_churn_prediction\churn_prediction\data_preprocess\data\repo_2\part_1')
#     saveUserActivePeriod(2, '2018-02-23', '2022-01-01',
#                          r'F:\MOOSE_cxy\developer_churn_prediction\churn_prediction\data_preprocess\data\repo_2\part_2')
#
#     params_list = [
#         ['churner',120,0.0],
#         ['churner',30,0.0],
#         ['loyaler',120,0.0],
#         ['loyaler', 120, 0.5],
#         ['loyaler', 30, 0.0],
#         ['loyaler', 30, 0.5]
#     ]
#     for param in params_list:
#         getModelUserPeriod(2, param[0],
#                            r'F:\MOOSE_cxy\developer_churn_prediction\churn_prediction\data_preprocess\data\repo_2\part_all',
#                            r'F:\MOOSE_cxy\developer_churn_prediction\churn_prediction\data_preprocess\data\repo_2\part_all\2_user_active_period.csv',
#                            period_length=param[1], overlap_ratio=param[2])
#         getModelUserPeriod(2, param[0],
#                            r'F:\MOOSE_cxy\developer_churn_prediction\churn_prediction\data_preprocess\data\repo_2\part_1',
#                            r'F:\MOOSE_cxy\developer_churn_prediction\churn_prediction\data_preprocess\data\repo_2\part_1\2_user_active_period.csv',
#                            period_length=param[1], overlap_ratio=param[2])
#         getModelUserPeriod(2, param[0],
#                            r'F:\MOOSE_cxy\developer_churn_prediction\churn_prediction\data_preprocess\data\repo_2\part_2',
#                            r'F:\MOOSE_cxy\developer_churn_prediction\churn_prediction\data_preprocess\data\repo_2\part_2\2_user_active_period.csv',
#                            period_length=param[1], overlap_ratio=param[2])