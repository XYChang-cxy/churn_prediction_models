from data_preprocess.database_connect import *
import networkx as nx
from networkx.algorithms.centrality import betweenness_centrality


def getDeveloperCollaborationNetwork(repo_id,startDay,endDay):
    user_index = dict()
    i = 0

    '''results = getRepoDataFromTable(['user_id'], 'repo_commit_comment', repo_id, startDay, endDay, is_distinct=True)
    for result in results:
        if result[0] not in user_index.keys():
            user_index[result[0]] = i
            i += 1'''

    results = getRepoDataFromTable(['user_id'], 'repo_issue_comment', repo_id, startDay, endDay,'create_time',
                                   is_distinct=True)
    for result in results:
        if result[0] not in user_index.keys():
            user_index[result[0]] = i
            i += 1

    results = getRepoDataFromTable(['user_id'], 'repo_review_comment', repo_id, startDay, endDay, 'create_time',
                                   is_distinct=True)
    for result in results:
        if result[0] not in user_index.keys():
            user_index[result[0]] = i
            i += 1

    '''results = getRepoDataFromTable(['user_id'], 'repo_review', repo_id, startDay, endDay, is_distinct=True)
    for result in results:
        if result[0] not in user_index.keys():
            user_index[result[0]] = i
            i += 1'''

    DCN = []
    for j in range(i):
        DCN.append([])
        for k in range(i):
            DCN[j].append(0)

    issue_comment_users = dict()
    review_users = dict()
    review_comment_users = dict()
    commit_comment_users = dict()

    results = getRepoDataFromTable(['issue_number','user_id'], 'repo_issue_comment', repo_id, startDay, endDay,'create_time',
                                   is_distinct=True)
    for result in results:
        if result[0] in issue_comment_users.keys():
            users = issue_comment_users[result[0]]
        else:
            users = []
        users.append(result[1])
        issue_comment_users.update({result[0]:users})

    results = getRepoDataFromTable(['pull_id','user_id'], 'repo_review_comment', repo_id, startDay, endDay,'create_time',
                                   is_distinct=True)
    for result in results:
        if result[0] in review_comment_users.keys():
            users = review_comment_users[result[0]]
        else:
            users = []
        users.append(result[1])
        review_comment_users.update({result[0]: users})

    '''results = getRepoDataFromTable(['commit_id','user_id'], 'repo_commit_comment', repo_id, startDay, endDay, is_distinct=True)
    for result in results:
        if result[0] in commit_comment_users.keys():
            users = commit_comment_users[result[0]]
        else:
            users = []
        users.append(result[1])
        commit_comment_users.update({result[0]: users})'''

    '''results = getRepoDataFromTable(['pull_id','user_id'], 'repo_review', repo_id, startDay, endDay, is_distinct=True)
    for result in results:
        if result[0] in review_users.keys():
            users = review_users[result[0]]
        else:
            users = []
        users.append(result[1])
        review_users.update({result[0]: users})'''

    for key in issue_comment_users.keys():
        for user1 in issue_comment_users[key]:
            for user2 in issue_comment_users[key]:
                DCN[user_index[user1]][user_index[user2]] +=1
    '''for key in review_users.keys():
        for user1 in review_users[key]:
            for user2 in review_users[key]:
                DCN[user_index[user1]][user_index[user2]] +=1'''
    for key in review_comment_users.keys():
        for user1 in review_comment_users[key]:
            for user2 in review_comment_users[key]:
                DCN[user_index[user1]][user_index[user2]] +=1
    '''for key in commit_comment_users.keys():
        for user1 in commit_comment_users[key]:
            for user2 in commit_comment_users[key]:
                DCN[user_index[user1]][user_index[user2]] += 1'''
    for j in range(len(user_index)):
        DCN[j][j] = 0
    DCN0 = []   # 无权零阶矩阵
    for j in range(i):
        DCN0.append([])
        for k in range(i):
            if DCN[j][k] > 0:
                DCN0[j].append(1)
            else:
                DCN0[j].append(0)
    index_user = dict()
    for key in user_index.keys():
        index_user.update({user_index[key]: key})

    return DCN, DCN0, index_user, user_index


#获取DCN的节点的度数
def getDCNDegrees(user_index, DCN):
    user_degree = dict()
    for user in user_index.keys():
        wd = 0
        for d in DCN[user_index[user]]:
            if d > 0:
                wd += 1
        user_degree.update({user:wd})
    return user_degree


#获取DCN的节点加权度
def getDCNWeightedDegrees(user_index, DCN):
    user_degree = dict()
    for user in user_index.keys():
        wd = 0
        for d in DCN[user_index[user]]:
            wd+=d
        user_degree.update({user:wd})
    return user_degree


def getUserDCNWeightedDegrees(user_id,user_index,DCN):
    user_degree = getDCNWeightedDegrees(user_index,DCN)
    if user_id in user_degree.keys():
        return user_degree[user_id]
    else:
        return 0


# 获取节点的介数中心性
def getBetweeness(DCN,user_index,index_user,normalized=True):
    G = nx.Graph()
    G.add_nodes_from(user_index.keys())
    for i in range(len(DCN)):
        for j in range(i + 1, len(DCN)):
            if DCN[i][j] > 0:
                G.add_edge(index_user[i], index_user[j])
    user_betweeness = betweenness_centrality(G, normalized=normalized)
    return user_betweeness


def getUserBetweeness(user_id,DCN,user_index,index_user,normalized=True):
    user_betweness = getBetweeness(DCN,user_index,index_user,normalized)
    if user_id in user_betweness.keys():
        return user_betweness[user_id]
    else:
        return 0
