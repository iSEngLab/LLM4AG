# encoding=utf-8

"""
Collect and compare the performance differences when generating from buggy versions and fixed versions.
"""

import os
from collections import OrderedDict, defaultdict
from typing import Dict

import argparse

from .metrics import *


def read_result_df(data_dir: str, src: str, idx: int, system: str, result_dir: str):
    result_file = os.path.join(data_dir, f"evosuite_{src}_tests/{idx}/{system}_generated/{result_dir}/full_test_data.csv")
    print(result_file)
    result_df = pd.read_csv(result_file)
    # deduplicate
    result_df.drop_duplicates(subset=["project", "bug_num",  "test_prefix"])
    return result_df


def cal_one_result(data_dir: str, src: str, idx: int, system: str, result_dir: str, scorers: Dict[str, Scorer]):
    result_df = read_result_df(data_dir, src, idx, system, result_dir)
    result = OrderedDict()
    for name, scorer in scorers.items():
        result[name] = scorer.score(result_df)
    return result


def cal_result(data_dir="data"):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    args = parser.parse_args()
    scorers = OrderedDict({
        "BugFound": BugFound(),
        "FPRate": FPR(),
        "Precision": Precision(),
        "TPs": TPs(),
        "FPs": FPs()
    })
    src_names = [ "buggy"]
    dfs = []
    all_dfs = defaultdict(dict)
    for src in src_names:
        for result_dir in ["results", "merged_results"]:
            exp_results = []
            num = 0
            for i in range(1, 11):
                try:
                    result = cal_one_result(data_dir, src, i, args.model_name, result_dir, scorers)
                    exp_results.append(result)
                    result_df = read_result_df(data_dir, src, i, args.model_name, result_dir)
                    condition = result_df.groupby(['project', 'bug_num']).TP.sum() > 0

                    # 使用条件来过滤DataFrame
                    filtered_df = result_df[
                        result_df.set_index(['project', 'bug_num']).index.isin(condition[condition].index)]
                    if not os.path.exists(args.model_name+'_'+result_dir):
                        os.mkdir(args.model_name+'_'+result_dir)
                    project = filtered_df['project']
                    bug_num = filtered_df['bug_num']
                    remains = []
                    for p,b in zip(project,bug_num):
                        remains.append('{}/{}'.format(p,b))
                    remains = set(remains)
                    f = open('{}/{}.txt'.format(args.model_name+'_'+result_dir,i),'a')
                    for r in remains:
                        f.write(r+'\n')
                    f.close()
                    num+=len(remains)
                except:
                    print(1)
                    continue
            print(num/10)
            src_df = pd.DataFrame(exp_results)
            all_dfs[src][result_dir] = src_df
            dfs.append(src_df.mean())
    indexes = [ "TEval@buggy_overfiltering", "TEval@buggy"]
    df = pd.DataFrame(dfs, index=indexes)
    print(df)
    print("dump result")
    df.to_csv('{}_rq1_2.csv'.format(args.model_name))


if __name__ == "__main__":
    cal_result()
