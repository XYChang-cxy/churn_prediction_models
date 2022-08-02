# 该文件用于获取模型训练和验证的详细数据，并存储到文件中
from data_preprocess.database_connect import *
import datetime
import numpy as np
import pandas as pd
from data_preprocess.developer_collaboration_network import *


# 获取数量相关指标的数据并存储
# id:仓库序号（1~30）
# user_period_file: 存储用户观察区间的文件
# period_length: 120/90/60/30,和user_period_file对应
# count_type: 数据类型，可选项包括：issue、issue comment、pull、pull merged、review、review comment、commit
def getCountDataAndSave(repo_id,user_period_file,period_length,count_type,save_dir):
    time_names = {
        'issue':'create_time',
        'issue comment':'create_time',
        'pull':'create_time',
        'pull merged':'merge_time',
        'review':'submit_time',
        'review comment':'create_time',
        'commit':'commit_time'
    }
    if period_length == 120 or period_length == 90 or period_length == 60:
        step = 10
    elif period_length == 30:
        step = 5
    else:
        print('period length error!')
        return
    if count_type not in time_names.keys():
        print('Data type error:',count_type+' not exist!')
        return

    user_periods = []
    with open(user_period_file,'r',encoding='utf-8')as f:
        f.readline()
        for line in f.readlines():
            items = line.strip(',\n').split(',')
            user_periods.append([int(items[0]),items[1],items[2]])
    f.close()

    user_period_file = user_period_file.replace('\\', '/')
    user_type = user_period_file.split('/')[-1].split('_')[1]
    tmp_str = user_period_file[user_period_file.rfind('/') + 1:]

    filename = save_dir + '/' + user_type + '_' + count_type.replace(' ', '_') \
               + tmp_str[tmp_str.find('-'):]
    print(filename)
    table_name = 'repo_' + count_type.replace(' ', '_')
    time_name = time_names[count_type]

    with open(filename,'w',encoding='utf-8')as f:
        line = 'user_id,'
        for i in range(int(period_length/step)):
            line+=str(i)+','
        f.write(line+'\n')
    f.close()

    for index in range(len(user_periods)):
        print(index, '/', len(user_periods))
        user_period = user_periods[index]
        user_id = user_period[0]
        startDay = user_period[1]
        endDay = user_period[2]
        count_list = []
        for i in range(int(period_length/step)):
            start_day = (datetime.datetime.strptime(startDay,fmt_day)+datetime.timedelta(days=i*step)).strftime(fmt_day)
            end_day = (datetime.datetime.strptime(start_day,fmt_day)+datetime.timedelta(days=step)).strftime(fmt_day)
            count = getUserDataFromTable(['count'],table_name,repo_id,user_id,start_day,end_day,time_name)
            count_list.append(count)
        line = str(user_id)+','
        for count in count_list:
            line += str(count)+','
        with open(filename,'a',encoding='utf-8')as f:
            f.write(line+'\n')
        f.close()


# 获取开发者在协作网络中的介数中心性或节点加权度
# repo_id: 仓库id
# user_period_file: 存储用户观察区间的文件
# period_length: 120/90/60/30,和user_period_file对应
# data_type: 数据类型，可选项包括：betweeness、weighted degree
# save_dir: 存储数据文件夹
def getDCNDataAndSave(repo_id,user_period_file,period_length,data_type,save_dir):
    if period_length == 120 or period_length == 90 or period_length == 60:
        step = 10
    elif period_length == 30:
        step = 5
    else:
        print('period length error!')
        return
    if data_type!='betweeness' and data_type!='weighted degree':
        print('Data type error: '+data_type+' not exist!')
        return

    user_periods = []
    with open(user_period_file, 'r', encoding='utf-8')as f:
        f.readline()
        for line in f.readlines():
            items = line.strip(',\n').split(',')
            user_periods.append([int(items[0]), items[1], items[2]])
    f.close()

    user_period_file = user_period_file.replace('\\', '/')
    user_type = user_period_file.split('/')[-1].split('_')[1]
    tmp_str = user_period_file[user_period_file.rfind('/') + 1:]

    filename = save_dir + '/' + user_type + '_' + data_type.replace(' ','_') \
               + tmp_str[tmp_str.find('-'):]
    print(filename)
    with open(filename,'w',encoding='utf-8')as f:
        line = 'user_id,'
        for i in range(int(period_length/step)):
            line+=str(i)+','
        f.write(line+'\n')
    f.close()

    for index in range(len(user_periods)):
        print(index,'/',len(user_periods))
        user_period = user_periods[index]
        user_id = user_period[0]
        startDay = user_period[1]
        endDay = user_period[2]
        value_list = []
        for i in range(int(period_length / step)):
            start_day = (datetime.datetime.strptime(startDay, fmt_day) + datetime.timedelta(days=i * step)).strftime(
                fmt_day)
            end_day = (datetime.datetime.strptime(start_day, fmt_day) + datetime.timedelta(days=step)).strftime(fmt_day)
            DCN, DCN0, index_user, user_index = getDeveloperCollaborationNetwork(repo_id,start_day,end_day)
            if data_type == 'betweeness':
                value = getUserDCNWeightedDegrees(user_id,user_index,DCN)
            else:
                value = getUserBetweeness(user_id,DCN,user_index,index_user,True)
            value_list.append(value)
        line = str(user_id) + ','
        for value in value_list:
            line += str(value) + ','
        with open(filename, 'a', encoding='utf-8')as f:
            f.write(line + '\n')
        f.close()


# 获取开发者在协作网络中的介数中心性和节点加权度
# repo_id: 仓库id
# user_period_file: 存储用户观察区间的文件
# period_length: 120/90/60/30,和user_period_file对应
# data_type_list: 数据类型，可选项包括：betweeness、weighted degree、betweeness和weighted degree
# save_dir: 存储数据文件夹
def getDCNDataAndSave2(repo_id,user_period_file,period_length,data_type_list,save_dir):
    if period_length == 120 or period_length == 90 or period_length == 60:
        step = 10
    elif period_length == 30:
        step = 5
    else:
        print('period length error!')
        return
    if 'betweeness' not in data_type_list and 'weighted degree' not in data_type_list:
        return

    user_periods = []
    with open(user_period_file, 'r', encoding='utf-8')as f:
        f.readline()
        for line in f.readlines():
            items = line.strip(',\n').split(',')
            user_periods.append([int(items[0]), items[1], items[2]])
    f.close()

    user_period_file = user_period_file.replace('\\', '/')
    user_type = user_period_file.split('/')[-1].split('_')[1]
    tmp_str = user_period_file[user_period_file.rfind('/') + 1:]

    filename1 = save_dir + '/' + user_type + '_' + data_type_list[0].replace(' ','_') \
               + tmp_str[tmp_str.find('-'):]
    print(filename1)
    with open(filename1,'w',encoding='utf-8')as f:
        line = 'user_id,'
        for i in range(int(period_length/step)):
            line+=str(i)+','
        f.write(line+'\n')
    f.close()

    if len(data_type_list) > 1:
        filename2 = save_dir + '/' + user_type + '_' + data_type_list[1].replace(' ', '_') \
                    + tmp_str[tmp_str.find('-'):]
        print(filename2)
        with open(filename2, 'w', encoding='utf-8')as f:
            line = 'user_id,'
            for i in range(int(period_length / step)):
                line += str(i) + ','
            f.write(line + '\n')
        f.close()

    for index in range(len(user_periods)):
        print(index,'/',len(user_periods))
        user_period = user_periods[index]
        user_id = user_period[0]
        startDay = user_period[1]
        endDay = user_period[2]
        value_list_1 = []
        value_list_2 = []
        for i in range(int(period_length / step)):
            start_day = (datetime.datetime.strptime(startDay, fmt_day) + datetime.timedelta(days=i * step)).strftime(
                fmt_day)
            end_day = (datetime.datetime.strptime(start_day, fmt_day) + datetime.timedelta(days=step)).strftime(fmt_day)
            DCN, DCN0, index_user, user_index = getDeveloperCollaborationNetwork(repo_id,start_day,end_day)
            data_type_1 = data_type_list[0]
            if data_type_1 == 'betweeness':
                value = getUserDCNWeightedDegrees(user_id,user_index,DCN)
            else:
                value = getUserBetweeness(user_id,DCN,user_index,index_user,True)
            value_list_1.append(value)
            if len(data_type_list)>1:
                data_type_2 = data_type_list[1]
                if data_type_2 == 'betweeness':
                    value = getUserDCNWeightedDegrees(user_id, user_index, DCN)
                else:
                    value = getUserBetweeness(user_id, DCN, user_index, index_user, True)
                value_list_2.append(value)

        line = str(user_id) + ','
        for value in value_list_1:
            line += str(value) + ','
        with open(filename1, 'a', encoding='utf-8')as f:
            f.write(line + '\n')
        f.close()

        if len(data_type_list)>1:
            line = str(user_id) + ','
            for value in value_list_2:
                line += str(value) + ','
            with open(filename2, 'a', encoding='utf-8')as f:
                f.write(line + '\n')
            f.close()


# 获取开发者接受到的响应数据并存储
# repo_id: 仓库id
# user_period_file: 存储用户观察区间的文件
# period_length: 120/90/60/30,和user_period_file对应
# data_type: 数据类型，可选项包括：issue comment、review、review comment
def getReceivedDataAndSave(repo_id,user_period_file,period_length,data_type,save_dir):
    if period_length == 120 or period_length == 90 or period_length == 60:
        step = 10
    elif period_length == 30:
        step = 5
    else:
        print('period length error!')
        return
    if data_type!='issue comment' and data_type!='review' and data_type!='review comment':
        print('Data type error: '+data_type+' not exist!')
        return
    user_periods = []
    with open(user_period_file, 'r', encoding='utf-8')as f:
        f.readline()
        for line in f.readlines():
            items = line.strip(',\n').split(',')
            user_periods.append([int(items[0]), items[1], items[2]])
    f.close()

    user_period_file=user_period_file.replace('\\','/')
    user_type = user_period_file.split('/')[-1].split('_')[1]
    tmp_str = user_period_file[user_period_file.rfind('/')+1:]

    filename = save_dir + '/' + user_type + '_received_' + data_type.replace(' ','_') \
               + tmp_str[tmp_str.find('-'):]
    print(filename)
    with open(filename, 'w', encoding='utf-8')as f:
        line = 'user_id,'
        for i in range(int(period_length / step)):
            line += str(i) + ','
        f.write(line + '\n')
    f.close()

    create_time = getRepoInfoFromTable(repo_id, ['created_at'])[0][0:10]
    for index in range(len(user_periods)):
        print(index,'/',len(user_periods))
        user_period = user_periods[index]
        user_id = user_period[0]
        startDay = user_period[1]
        endDay = user_period[2]
        count_list = []
        for i in range(int(period_length / step)):
            start_day = (datetime.datetime.strptime(startDay, fmt_day) + datetime.timedelta(days=i * step)).strftime(
                fmt_day)
            end_day = (datetime.datetime.strptime(start_day, fmt_day) + datetime.timedelta(days=step)).strftime(fmt_day)
            if data_type == 'issue comment':
                results = getUserDataFromTable(['issue_number'],'repo_issue',repo_id,user_id,create_time,end_day,
                                               'create_time')
                issue_number_list = []
                for result in results:
                    issue_number_list.append(result[0])
                if len(issue_number_list)==0:
                    count_list.append(0)
                else:
                    results = getRepoDataFromTable(['issue_number'],'repo_issue_comment',repo_id,start_day,end_day,
                                                   'create_time')
                    count = 0
                    for result in results:
                        if result[0] in issue_number_list:
                            count += 1
                    count_list.append(count)
            else:
                results = getUserDataFromTable(['pull_id'],'repo_pull',repo_id,user_id,create_time,end_day,'create_time')
                pull_list = []
                for result in results:
                    pull_list.append(result[0])
                if len(pull_list)==0:
                    count_list.append(0)
                else:
                    table_name = 'repo_'+data_type.replace(' ','_')
                    if table_name=='repo_review':
                        time_name = 'submit_time'
                    else:
                        time_name = 'create_time'
                    results = getRepoDataFromTable(['pull_id'],table_name,repo_id,start_day,end_day,time_name)
                    count = 0
                    for result in results:
                        if result[0] in pull_list:
                            count += 1
                    count_list.append(count)
        line = str(user_id) + ','
        for count in count_list:
            line += str(count) + ','
        with open(filename, 'a', encoding='utf-8')as f:
            f.write(line + '\n')
        f.close()



# if __name__ == '__main__':
#     # getCountDataAndSave(2,r'C:\Users\cxy\Desktop\test\repo_loyalers_period-120-0.0.csv',120,'commit',
#     #                     r'C:\Users\cxy\Desktop\test')
#     # getDCNDataAndSave(2, r'C:\Users\cxy\Desktop\test\repo_loyalers_period-120-0.0.csv', 120, 'betweeness',
#     #                     r'C:\Users\cxy\Desktop\test')
#     getReceivedDataAndSave(2, r'C:\Users\cxy\Desktop\test\repo_loyalers_period-120-0.0.csv', 120, 'review comment',
#                       r'C:\Users\cxy\Desktop\test')