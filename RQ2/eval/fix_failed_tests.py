# encoding=utf-8

from collections import defaultdict
import glob
import logging
import os
import argparse
import re
import shutil
import subprocess
import pandas as pd
from typing import Dict, Set


def collect_failed_lines(error_msg: str) -> Dict[str, Set[int]]:
    """
    return Dict[filepath, line_no_set]
    """
    # we only care about error
    warning_pattern = r'.*?(\S*_ESTest\.java):(\d+):\s+warning:.*'
    error_pattern = r'.*?(\S*_ESTest\.java):(\d+):\s+error:.*'
    error_lines = re.split(r'\n+', error_msg)
    failed_lines = defaultdict(set)
    for line in error_lines:
        if "_ESTest.java" in line:
            match = re.match(error_pattern, line)
            if not match:
                if not re.match(warning_pattern, line):
                    logging.error(f"Can not find line no in: {line}")
                continue
            filepath = '/'.join(match.group(1).split('/')[4:])
            line_no = int(match.group(2))
            failed_lines[filepath].add(line_no)
    return failed_lines


def fix_aggregate_test(aggregated_dir, output_dir, method, project, bug_id, error_msg):
    method_sig_re = r'\s*public\s+void\s+(\S+)\(\)\s+throws\s+Throwable\s+{\s*'

    agg_test_dir = os.path.join(aggregated_dir, project, method, str(bug_id))
    out_test_dir = os.path.join(output_dir, project, method, str(bug_id))
    
    failed_lines = collect_failed_lines(error_msg)
    if len(failed_lines) == 0:
        logging.error(f"ERROR: No failed lines for {project} {bug_id}")
    
    # copy the aggregated test to another dir
    if os.path.exists(out_test_dir):
        subprocess.run(f'rm -r {out_test_dir}'.split())
    parent_dir = os.path.dirname(out_test_dir)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    shutil.copytree(agg_test_dir, out_test_dir)
    # rm tar.bz2
    tarball_name = None
    for tarball in glob.glob(f"{out_test_dir}/*.tar.bz2"):
        tarball_name = tarball.split('/')[-1]
        os.remove(tarball)

    for filepath, failed_line_nos in failed_lines.items():
        file_full_path = os.path.join(out_test_dir, filepath)

        with open(file_full_path, 'r') as f:
            code_lines = f.read().split('\n')

        method_names = []
        at_test_line_idxs = []
        last_bracket_line_idx = len(code_lines)
        for idx, line in enumerate(code_lines):
            if line.strip() == "}":
                last_bracket_line_idx = idx
            match = re.match(method_sig_re, line)
            if match:
                method_name = match.group(1)
                method_names.append(method_name)
                at_test_line_idxs.append(idx-1)
        at_test_line_idxs.append(last_bracket_line_idx)

        assert len(method_names)+1 == len(at_test_line_idxs)
        failed_line_idx = sorted(list(map(lambda x:x-1,failed_line_nos)))
        failed_line_idx_iter = iter(failed_line_idx)
        next_failed_line_idx = next(failed_line_idx_iter)

        # header 
        new_file_lines = []
        prev_line_idx = 0
        while next_failed_line_idx is not None and next_failed_line_idx < at_test_line_idxs[0]:
            line = code_lines[next_failed_line_idx]
            if line.startswith("import"):
                new_file_lines += code_lines[prev_line_idx:next_failed_line_idx]
                prev_line_idx = next_failed_line_idx+1
            else:
                pass
            try:
                next_failed_line_idx = next(failed_line_idx_iter)
            except:
                next_failed_line_idx = None
        new_file_lines += code_lines[prev_line_idx:at_test_line_idxs[0]]

        for i in range(len(method_names)):
            start, end = at_test_line_idxs[i], at_test_line_idxs[i+1]
            line_range_set = set(range(start, end))
            keep = True
            while next_failed_line_idx in line_range_set:
                keep = False
                try:
                    next_failed_line_idx = next(failed_line_idx_iter)
                except StopIteration:
                    next_failed_line_idx = None

            if keep:
                new_file_lines += code_lines[start:end]
        new_file_lines += code_lines[at_test_line_idxs[-1]:]
        
        os.remove(file_full_path)
        with open(file_full_path, 'w') as f:
            f.write('\n'.join(new_file_lines))

        cwd = os.getcwd()
        os.chdir(out_test_dir)
        assert tarball_name
        test_dir = filepath.split('/')[0]
        subprocess.run(f'tar cjf {tarball_name} {test_dir}'.split())
        os.chdir(cwd)


def main(result_dir: str,
         aggregated_dir: str,
         output_dir: str,
         method: str):
    failed_tests_df = pd.read_csv(os.path.join(result_dir, "failed_tests.csv")).fillna('')
    for test in failed_tests_df.itertuples(index=False):
        project, bug_id, error_msg = test.project, test.bug_id, test.error_msg
        logging.info(f"Fixing {project} {bug_id}")
        if not error_msg:
            continue
        fix_aggregate_test(aggregated_dir, output_dir, method, project, bug_id, error_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('aggregated_dir')
    parser.add_argument('result_dir')
    parser.add_argument('output_dir')
    parser.add_argument('-m', '--method', dest="method", type=str, default="toga")
    args = parser.parse_args()

    main(args.result_dir, args.aggregated_dir, args.output_dir, args.method)
