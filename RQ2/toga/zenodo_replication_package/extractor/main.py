# encoding=utf-8
import logging
import os
import shutil
from common.d4j import parse_projects
import fire
import pandas as pd
import subprocess as sp
from joblib import delayed, Parallel
from . import extract_tests
from tqdm import tqdm

"""
1. Generate tests from the buggy version
2. Construct inputs.csv and meta.csv using toga's script
"""


def setUpLog():
    # joblib will not copy the global state of a module
    logging.basicConfig(level=logging.INFO)


setUpLog()

# 3 minutes
BUDGET = 180
SKIP_BUGS = {
}


def gen_tests_for_bug(proj: str,
                      bug_num: int,
                      out_dir: str,
                      suffix: str,
                      seed: int):
    setUpLog()
    system = "evosuite"
    test_id = str(bug_num * 100)
    # for each bug, there can only have one tar.bz2
    # therefore, we just remove it and regenerate
    test_suite = os.path.join(out_dir, proj, system, test_id, f"{proj}-{bug_num}{suffix}-{system}.{test_id}.tar.bz2")
    if os.path.exists(test_suite):
        logging.warning(f"Skip gen_tests for {proj}-{bug_num}")
        return proj, bug_num, "skip"
    # following defects4j
    random_seed = seed * 1000 + bug_num
    cmd = f"gen_tests.pl -g {system} -p {proj} -v {bug_num}{suffix} -n {test_id} -o {out_dir} -b {BUDGET} " \
          f"-s {random_seed}"
    logging.info(f"run cmd: {cmd}")

    result = sp.run(cmd.split(), env=os.environ.copy(), stdout=sp.PIPE, stderr=sp.PIPE, text=True)
    if result.returncode != 0:
        logging.info(f"retry {cmd}")
        result = sp.run(cmd.split(), env=os.environ.copy(), stdout=sp.PIPE, stderr=sp.PIPE, text=True)
        if result.returncode != 0:
            logging.error(f"Run {cmd}:\nstdout: {result.stdout}\nstderr: {result.stderr}")
            return proj, bug_num, "fail"
    return proj, bug_num, "pass"


def skip_bug(proj: str, bug_num: int):
    # useless
    if bug_num in SKIP_BUGS.get(proj, set()):
        logging.info(f"Force skipping gen_tests for {proj}-{bug_num}")
        return True
    return False


def gen_tests_for_proj(proj: str,
                       meta_file: str,
                       out_dir: str,
                       suffix: str,
                       seed: int):
    meta = pd.read_csv(meta_file, header=0)
    for bug_num in meta["bug.id"]:
        bug_num = int(bug_num)
        if skip_bug(proj, bug_num):
            continue
        gen_tests_for_bug(proj, bug_num, out_dir, suffix, seed)


def gen_tests(seed: int,
              proj: str = None,
              meta_dir: str = "data/metadata",
              out_dir: str = "data/evosuite_buggy_regression_all/generated",
              suffix: str = "b",
              n_jobs: int = None):
    projs = parse_projects(proj)

    tasks = []
    for proj in projs:
        meta_file = os.path.join(meta_dir, f"{proj}.csv")
        meta = pd.read_csv(meta_file, header=0)
        # for each active bug
        for bug_num in meta["bug.id"]:
            bug_num = int(bug_num)
            if skip_bug(proj, bug_num):
                continue
            # run evosuite to generate tests
            tasks.append(delayed(gen_tests_for_bug)(proj, bug_num, out_dir, suffix, seed))
    results = Parallel(n_jobs=n_jobs, prefer='processes')(tqdm(tasks))
    df = pd.DataFrame(results, columns=["project", "bug_id", "success"])
    result_file = os.path.join(out_dir, "gen_tests_result.csv")
    df.to_csv(result_file, index=False)


def prepare_tests(test_dir: str = "data/evosuite_buggy_regression_all/generated"):
    system = "evosuite"
    projs = parse_projects(None)

    for proj in projs:
        base_dir = os.path.join(test_dir, proj, system)
        for dir_name_str in os.listdir(base_dir):
            dir_name = int(dir_name_str)
            if dir_name >= 100:
                new_name = str(dir_name // 100)
                shutil.move(os.path.join(base_dir, dir_name_str), os.path.join(base_dir, new_name))

        for dirpath, dirnames, filenames in os.walk(base_dir):
            for file in filenames:
                if file.endswith(".tar.bz2"):
                    sp.run(f"cd {dirpath};tar xf {file}", shell=True, check=True)


def ex_tests(test_corpus_dir: str = "data/evosuite_buggy_regression_all",
             output_dir: str = "data/evosuite_buggy_tests",
             sample_5projects: bool = False,
             d4j_path: str = "/data/swf/defects4j",
             njobs: int = None):
    d4j_path = os.path.expanduser(d4j_path)
    extract_tests.main(test_corpus_dir, sample_5projects, d4j_path, output_dir, njobs)


if __name__ == "__main__":
    fire.Fire({
        'gen_tests': gen_tests,
        'prepare_tests': prepare_tests,
        'ex_tests': ex_tests
    })
