# encoding=utf-8

"""
Merge the test results of the first run and the fixed run
"""

import logging
import os
import glob
import argparse
import shutil
import pandas as pd
from pandas import DataFrame
from collections import OrderedDict


def merge_bug_detection(agg_result_dir, fixed_result_dir, out_result_dir):
    agg_bd_df = pd.read_csv(os.path.join(agg_result_dir, "bug_detection"), dtype=str).fillna('')
    fixed_bd_df = pd.read_csv(os.path.join(fixed_result_dir, "bug_detection"), dtype=str).fillna('')

    def _build_bd_dict(bd_df: DataFrame):
        result_dict = OrderedDict()
        for row in bd_df.itertuples():
            key = "-".join(row[1:-2])
            # remove index and convert to tuple
            result_dict[key] = row[1:]
        return result_dict

    agg_bd_dict = _build_bd_dict(agg_bd_df)
    fixed_bd_dict = _build_bd_dict(fixed_bd_df)
    
    for key, row in fixed_bd_dict.items():
        if key in agg_bd_dict:
            # replace the old row with the new one
            agg_bd_dict[key] = row
    
    new_df = DataFrame(list(agg_bd_dict.values()), columns=agg_bd_df.columns)
    if os.path.exists(out_result_dir):
        shutil.rmtree(out_result_dir)
    os.makedirs(out_result_dir)
    new_df.to_csv(os.path.join(out_result_dir, "bug_detection"), index=False,encoding='utf-8')


def merge_bug_detection_log(agg_result_dir, fixed_result_dir, out_result_dir):
    out_bd_log_dir = os.path.join(out_result_dir, "bug_detection_log")
    if os.path.exists(out_bd_log_dir):
        shutil.rmtree(out_bd_log_dir)
    agg_bd_log_dir = os.path.join(agg_result_dir, "bug_detection_log")
    shutil.copytree(agg_bd_log_dir, out_bd_log_dir)
    fixed_bd_log_dir = os.path.join(fixed_result_dir, "bug_detection_log")
    
    for log_file in glob.glob(f"{fixed_bd_log_dir}/*/*/*.trigger.log"):
        project, method, filename = log_file.split('/')[-3:]
        out_dir = os.path.join(out_bd_log_dir, project, method)
        out_file = os.path.join(out_dir, filename)
        if os.path.exists(out_file):
            logging.warning(f"{out_file} exists")
            items = filename.split('.')
            tag = items[0][-1]
            another_tag = 'b' if tag == 'f' else 'f'
            another_filename = ".".join([items[0][:-1]+another_tag] + items[1:])
            another_file = os.path.join(agg_bd_log_dir, project, method, another_filename)
            if os.path.exists(another_file):
                logging.error(f"{out_file} exists!")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        shutil.copy(log_file, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="agg_result_dir")
    parser.add_argument(dest="fixed_result_dir")
    parser.add_argument(dest="out_result_dir")
    args = parser.parse_args()

    agg_result_dir = args.agg_result_dir
    fixed_result_dir = args.fixed_result_dir
    out_result_dir = args.out_result_dir

    if os.path.exists(out_result_dir):
        shutil.rmtree(out_result_dir)
    os.makedirs(out_result_dir)

    merge_bug_detection(agg_result_dir, fixed_result_dir, out_result_dir)
    merge_bug_detection_log(agg_result_dir, fixed_result_dir, out_result_dir)
