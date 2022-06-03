# 该文件用于获取开发者的留存时间段（等于活动时间段+流失期限时间），或测试一段时间内用户每周有无活动情况
from data_preprocess.get_user import *


def getFirstAndLast(repo_id,user_id,churn_limit_weeks,start_time=''):
    if start_time !='':
        create_time = start_time
    else:
        create_time = getRepoInfoFromTable(repo_id, ['created_at'])[0][0:10]

    end_time = datetime.datetime.now().strftime(fmt_day)
    print(create_time,end_time)
    time_delta = (datetime.datetime.strptime(end_time,fmt_day)-datetime.datetime.strptime(create_time,fmt_day)).days
    period_weeks = int(time_delta/7)
    flag=False
    left_list = []
    right_list = []
    churn_limit = churn_limit_weeks

    for i in range(period_weeks):
        start_day = (datetime.datetime.strptime(create_time,fmt_day)+datetime.timedelta(days=i*7)).strftime(fmt_day)
        end_day = (datetime.datetime.strptime(start_day,fmt_day)+datetime.timedelta(days=7)).strftime(fmt_day)
        print(i,'/',period_weeks,start_day,end_day)
        list = getRepoUserList(repo_id, start_day, end_day)
        if user_id in list:
            churn_limit = churn_limit_weeks
            if flag == False:
                flag=True
                print('left:',start_day)
                left_list.append(start_day)
        if flag and user_id not in list:
            churn_limit-=1
            if churn_limit <= 0:
                flag = False
                churn_limit = churn_limit_weeks
                print('right:',end_day)
                right_list.append(end_day)
    if len(right_list)<len(left_list):
        right_list.append(end_time)
    ret=[]
    for i in range(len(left_list)):
        ret.append(left_list[i]+'--'+right_list[i])
    return ret


def testUserPeriod(repo_id,user_id,startDay,endDay):
    time_delta = (datetime.datetime.strptime(endDay, fmt_day) - datetime.datetime.strptime(startDay, fmt_day)).days
    period_weeks = int(time_delta / 7)
    true_count=0
    false_count=0
    for i in range(period_weeks):
        start_day = (datetime.datetime.strptime(startDay, fmt_day) + datetime.timedelta(days=i * 7)).strftime(
            fmt_day)
        end_day = (datetime.datetime.strptime(start_day, fmt_day) + datetime.timedelta(days=7)).strftime(fmt_day)
        print(i, '/', period_weeks, start_day, end_day)
        list = getRepoUserList(repo_id, start_day, end_day)
        if user_id in list:
            print('True',end=' ')
            false_count = 0
            true_count += 1
            print(true_count)
        else:
            print('False',end=' ')
            true_count = 0
            false_count += 1
            print(false_count)



if __name__ == '__main__':
    repo_id = 8649239
    user_id = 7854183
    # print(getFirstAndLast(repo_id,user_id,14,'2021-10-28'))

    startDay='2021-10-28'
    endDay='2022-06-03'
    # testUserPeriod(repo_id,user_id,startDay,endDay)
