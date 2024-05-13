# encoding=utf-8

from collections import OrderedDict

import fire
import numpy as np

from .metrics import *
from .rq1_2 import cal_one_result, read_result_df


def cal_result(data_dir="data"):
    # ten experiments
    scorers = OrderedDict({
        "BugFound": BugFound(),
        "Precision": Precision(),
        "TPs": TPs(),
        "FPs": FPs()
    })
    dfs = []
    all_dfs = {}
    src = "buggy"
    for system in ["naive", "toga"]:
        exp_results = []
        for i in range(1, 11):
            result = cal_one_result(data_dir, src, i, system, "merged_results", scorers)
            exp_results.append(result)
        assert len(exp_results) == 10
        src_df = pd.DataFrame(exp_results)
        all_dfs[system] = src_df
        dfs.append(src_df.mean())
    indexes = ["NoException", "TOGA"]
    df = pd.DataFrame(dfs, index=indexes)
    print(df)
    print("dump result")
    df.to_csv('results/rq3.csv')


def TP_bug(data_dir: str, src: str, idx: int, system: str, result_dir: str):
    df = read_result_df(data_dir, src, idx, system, result_dir)
    grouped_data = df.groupby(['project', 'bug_num'], as_index=False)
    df2 = grouped_data.TP.sum()
    TP = df2[df2.TP > 0]
    assert len(TP) == (grouped_data.TP.sum().TP > 0).sum()
    TP_bugs = TP[["project", "bug_num"]]
    return TP_bugs


def cal_intersection(data_dir="data"):
    src = "buggy"
    inter_nums = []
    for i in range(1, 11):
        naive_TP_bugs = TP_bug(data_dir, src, i, "naive", "merged_results")
        toga_TP_bugs = TP_bug(data_dir, src, i, "toga", "merged_results")
        inter_df = pd.merge(naive_TP_bugs, toga_TP_bugs, how='inner')
        inter_nums.append(len(inter_df))
    print(inter_nums)
    print(np.array(inter_nums).mean())


if __name__ == "__main__":
    fire.Fire({
        "cal_result": cal_result,
        "cal_intersection": cal_intersection
    })