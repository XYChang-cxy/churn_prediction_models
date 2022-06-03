# from open_search import get_index_content
# import data_transformation
import json


f = open('new_json_2.json', 'r')
content = f.read()
hashmap = json.loads(content)
f.close()


def get_table(table_name, day_start_before, day_end_before):
    return hashmap[table_name]
    if table_name == 'churn_search_repos_final':
        # Get the related data from OpenSeaerch
        churn_search_repos_final = get_index_content('gitee_repo-raw', day_start_before, day_end_before, ["data.full_name", "data.id", "data.created_at"])
        # Raw data schema transformation
        churn_search_repos_final = data_transformation.churn_search_repos_final_format(churn_search_repos_final)

        return churn_search_repos_final

    elif table_name == 'repo_issue' or table_name == 'repo_issue_comment':
        repo_issue_and_repo_issue_comment = get_index_content('gitee_issues-raw', day_start_before, day_end_before, ["data.repository.id", "data.id", "data.number", "data.created_at", "data.user.id", "data.issue_state", "data.comments_data.id", "data.comments_data.created_at", "data.comments_data.user.id"])  # author_association，该字段可暂时不使用，但我这里要留个位置。
        repo_issue, repo_issue_comment = data_transformation.repo_issue_and_repo_issue_comment_format(repo_issue_and_repo_issue_comment)

        if table_name == 'repo_issue':
            return repo_issue
        else:
            return repo_issue_comment

    elif table_name == 'repo_pull' or table_name == 'repo_pull_merged' or table_name == 'repo_review_comment':
        repo_pull_and_repo_pull_merged_and_repo_review_comment = get_index_content('gitee_pulls-raw', day_start_before, day_end_before, ["data.base.repo.id", "data.id", "data.number", "data.created_at", "data.merged_at", "data.state", "data.user.id", "data.review_comments_data.id", "data.review_comments_data.created_at", "data.review_comments_data.user.id"])
        repo_pull, repo_pull_merged, repo_review_comment = data_transformation.repo_pull_and_repo_pull_merged_and_repo_review_comment_format(repo_pull_and_repo_pull_merged_and_repo_review_comment)

        if table_name == 'repo_pull':
            return repo_pull
        elif table_name == 'repo_pull_merged':
            return repo_pull_merged
        else:
            return repo_review_comment
    
    # repo_review = # Gitee里似乎没有review数据，暂时放弃这部分数据，模型训练时也可以先剔除这部分数据
    # repo_commit # 在OpenSearch中未找到，暂时不使用
    # repo_commit_comment # 在OpenSearch中未找到，暂时不使用
    # repo_star # 模型训练和预测暂时不需要repo_star数据
    # repo_fork # 模型训练和预测暂时不需要repo_fork数据
    # user_data # user_data的数据暂时和模型无关，但通过此表可以将user_id对应到具体的用户login

    return None

if __name__ == '__main__':
    day_start_before = 30
    day_end_before = 7
    churn_search_repos_final = get_table('churn_search_repos_final', day_start_before, day_end_before) # get_table 方法在 util.py 文件中, 需提前 from relative_path/util import get_table
    repo_issue = get_table('repo_issue', day_start_before, day_end_before)
    repo_issue_comment = get_table('repo_issue_comment', day_start_before, day_end_before)
    repo_pull = get_table('repo_pull', day_start_before, day_end_before)
    repo_pull_merged = get_table('repo_pull_merged', day_start_before, day_end_before)
    repo_review_comment = get_table('repo_review_comment', day_start_before, day_end_before)

    print('churn_search_repos_final: {} records in total from {} days before to {} days before'.format(len(churn_search_repos_final), day_start_before, day_end_before))
    print(churn_search_repos_final[0].keys())
    print(churn_search_repos_final[:2])
    print('\n')

    print('repo_issue: {} records in total from {} days before to {} days before'.format(len(repo_issue), day_start_before, day_end_before))
    print(repo_issue[0].keys())
    print(repo_issue[:2])
    print('\n')
    print('repo_issue_comment: {} records in total from {} days before to {} days before'.format(len(repo_issue_comment), day_start_before, day_end_before))
    print(repo_issue_comment[0].keys())
    print(repo_issue_comment[:2])
    print('\n')

    print('repo_pull: {} records in total from {} days before to {} days before'.format(len(repo_pull), day_start_before, day_end_before))
    print(repo_pull[0].keys())
    print(repo_pull[:2])
    print('\n')
    print('repo_pull_merged: {} records in total from {} days before to {} days before'.format(len(repo_pull_merged), day_start_before, day_end_before))
    print(repo_pull_merged[0].keys())
    print(repo_pull_merged[:2])
    print('\n')
    print('repo_review_comment: {} records in total from {} days before to {} days before'.format(len(repo_review_comment), day_start_before, day_end_before))
    print(repo_review_comment[0].keys())
    print(repo_review_comment[:2])
    print('\n')