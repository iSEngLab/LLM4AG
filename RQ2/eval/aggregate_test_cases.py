from itertools import chain
import logging
import time
import os, re, argparse
from subprocess import run
from pathlib import Path
from collections import defaultdict
from glob import glob
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd


def close_test_harnesses(test_base_dir):
    test_harness_fs = Path(test_base_dir).rglob('*ESTest.java')

    for test_harness_f in test_harness_fs:
        with open(test_harness_f, 'a') as f:
            f.write('\n}')


def get_imports(java_file):
    imports = []
    with open(java_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('import'):
                imports += [line]
            if line.startswith('@RunWith'):
                break

    return imports


def get_test_harness(test_case_dir, output_test_dir):
    cwd = os.getcwd()
    try:
        os.chdir(test_case_dir)
        # run(f'tar -xf test_case.tar.bz2'.split())

        test_file = str(list(Path('.').rglob('*ESTest.java'))[0])

        with open('./test.txt') as f:
            test_case = f.read()

        # if harness not created yet, make it
        if not os.path.isfile(output_test_dir+'/'+test_file):
            test_dir = test_file.split('/')[0]
            run(f'cp -r {test_dir} {output_test_dir}'.split())

            with open(test_file) as f:
                test_file_txt = f.read()

            test_harness = test_file_txt.split(test_case)
            header = test_harness[0]

            # write first half of test harness (needs '}' at end later)
            with open(output_test_dir+'/'+test_file, 'w') as f:
                f.write(header)

        # if harness exists, check imports match
        else:
            existing_imports = set(get_imports(output_test_dir+'/'+test_file))
            new_imports = set(get_imports(test_file))

            assert new_imports == existing_imports

    except AssertionError as e:
        raise e
    except Exception as e:
        logging.error('ERROR:', e, test_case_dir)
        raise e
    finally:
        os.chdir(cwd)
        pass

    return test_file, test_case


def aggregate_bug_tests(project, bug, bug_dir, aggregate_bug_dir, generator: str):
    cwd = os.getcwd()

    test_id_to_uniq_name = []

    try:
        if os.path.isdir(aggregate_bug_dir):
            run(f'rm -r {aggregate_bug_dir}'.split())
        os.makedirs(aggregate_bug_dir)
        os.chdir(aggregate_bug_dir)
        
        test_name_re = re.compile(r'(\s*@Test.*public void )(\w+)(\(\).+)', re.DOTALL)

        test_names = defaultdict(lambda: defaultdict(lambda: 0))

        test_file = None
        ntests_collected = 0

        test_case_dirs = list(glob(bug_dir+'/*'))
        test_case_dirs =  sorted(test_case_dirs, key=lambda x:int(os.path.basename(x)))

        for test_case_dir in test_case_dirs: 
            test_id = os.path.basename(test_case_dir)
            test_file, test_case = get_test_harness(test_case_dir, os.getcwd())
            
            # get test case name
            match = test_name_re.match(test_case)
            if match is None:
                logging.error(f'ERROR could not match test:\n{test_case}')
                continue
            test_pre, test_name, test_post = match.groups()
            
            # add to dict with number
            test_names[test_file][test_name] += 1
            
            # get test with unique name
            test_name_uniq = test_name + str(test_names[test_file][test_name])
            full_uniq_name = test_file.replace('/', '.')[:-5] + "::" + test_name_uniq
            test_id_to_uniq_name.append((project, bug, test_id, full_uniq_name))
            test_uniq = test_pre + test_name_uniq + test_post

            # append to aggregate test file
            with open(test_file, 'a') as f:
                f.write('\n'+test_uniq+'\n')

            ntests_collected += 1

        if test_file:
            test_base_dir = test_file.split('/')[0]
            close_test_harnesses(test_base_dir)

            tarball_name = f'{project}-{bug}b-{generator}.{bug}.tar.bz2'
            run(f'tar cjf {tarball_name} {test_base_dir}'.split())

            logging.info(f'collected {ntests_collected} for {bug_dir}')

        else:
            logging.info(f'no tests found for {bug_dir}')
            
    except AssertionError as e:
        raise e
    except Exception as e:
        logging.error('ERROR:', e, bug_dir)
        raise e
    finally:
        os.chdir(cwd)

    return test_id_to_uniq_name    


def aggregate_all_project_tests(all_projects_dir, aggregate_dir, generator: str, verbose: bool = False):
    task_args = []
    for project_dir in glob(all_projects_dir+'/*'):
        project = os.path.basename(project_dir)
        logging.info(project_dir)

        for bug_dir in glob(f"{project_dir}/{generator}/*"):
            bug = os.path.basename(bug_dir)
            logging.info(bug)

            aggregate_bug_dir = f'{aggregate_dir}/{project}/{generator}/{bug}'

            task_args.append(
                (project, bug, bug_dir, aggregate_bug_dir, generator)
            )

    tasks = [delayed(aggregate_bug_tests)(*task_arg) for i, task_arg in enumerate(task_args)]
    
    if verbose:
        n_jobs = None
    else:
        n_jobs = -1

    results = Parallel(n_jobs=n_jobs, prefer='processes')(tqdm(tasks))
    return list(chain(*results))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_path')
    parser.add_argument('-g', '--generator', type=str, default="toga")
    parser.add_argument('-v', action='store_true')
    args = parser.parse_args()

    CORPUS_DIR = os.getcwd() + '/'+args.corpus_path 
    all_projects_dir = CORPUS_DIR + '/generated_d4j_tests'
    aggregate_dir = CORPUS_DIR + '/aggregated_d4j_tests'
    generator = args.generator
    if generator not in {"toga", "naive"}:
        raise Exception("Unknown generator: should be toga or naive")

    start_time = time.time()
    test_id_to_name_uniq = aggregate_all_project_tests(all_projects_dir, aggregate_dir, generator, verbose=args.v)
    
    df = pd.DataFrame(test_id_to_name_uniq, columns=["project", "bug_num", "test_id", "unique_test_name"])
    df.to_csv(os.path.join(CORPUS_DIR, "test_id_to_name_uniq.csv"), index=False,encoding='utf-8')
    df["bug_num"] = df["bug_num"].astype(int)
    df["test_id"] = df["test_id"].astype(int)

    # build the mapping from test_id to unique_test_name
    test_id_df = pd.read_csv(os.path.join(CORPUS_DIR, "test_ids.csv")) 
    id_to_name_df = test_id_df.merge(df, how="outer", on=["project", "bug_num", "test_id"], validate="one_to_one")
    id_to_name_df.to_csv(os.path.join(CORPUS_DIR, "uniq_test_names.csv"), index=False,encoding='utf-8')
    print(time.time() - start_time)
