import logging
from joblib import Parallel, delayed
import glob, os, argparse
import subprocess
from tqdm import tqdm
import pandas


def run(project, gen_dir, result_dir, timeout, args, i, tot):
    cmd=f'run_bug_detection.pl -p {project} -d {gen_dir} -o {result_dir}'
    success = True 
    is_time_out = False
    error_msg = None
    if args.v:
        logging.info(f'running ({i}/{tot}):', cmd)
    try:
        res = subprocess.run(cmd.split(), capture_output=True, timeout=timeout)
        stdout = res.stdout.decode('utf-8')
        stderr = res.stderr.decode('utf-8')
        if 'FAILED' in stdout or 'FAILED' in stderr:
            success = False
            error_msg = stderr
            logging.error(f'failed: {cmd}')
            logging.error(stdout)
            logging.error(stderr)
        elif args.v:
            logging.info(stdout)
            logging.info(stderr)
    except subprocess.TimeoutExpired:
        is_time_out = True

    return gen_dir, success, is_time_out, error_msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gen_test_dir')
    parser.add_argument('-o', dest='result_dir', default='results')
    parser.add_argument('-t', '--timeout', dest='timeout', type=int, default=180)
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-n', '--njobs', dest='njobs', type=int, default=None)
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()

    gen_test_dir = args.gen_test_dir
    result_dir = args.result_dir
    njobs = args.njobs

    if os.path.exists(result_dir):
        subprocess.run(f'rm -r {result_dir}'.split())
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    projects_dirs = glob.glob(gen_test_dir + '/*')

    task_args = []
    for pd in projects_dirs:
        p = os.path.basename(pd)

        for gen_dir in glob.glob(f'{pd}/*/*'):
            if glob.glob(gen_dir+'/*.tar.bz2'):
                task_args += [(p, gen_dir, result_dir)]

    tot = len(task_args)
    if args.test:
        task_args = task_args[:32]
    tasks = [delayed(run)(*task_arg, args.timeout, args, i+1, tot) for i, task_arg in enumerate(task_args)]

    if not args.v:
        results = Parallel(n_jobs=njobs, prefer='processes')(tqdm(tasks))
    else:
        results = Parallel(n_jobs=None, prefer='processes')(tqdm(tasks))

    failed_tests = []
    # eval failed test
    for result in results:
        cur_gen_dir, success, is_time_out, error_msg = result
        if success and not is_time_out:
            continue
        items = cur_gen_dir.split('/')
        project, method, bug_id = items[-3:]
        item = [project, bug_id, error_msg]
        failed_tests.append(item)
    
    failed_tests_df = pandas.DataFrame(failed_tests, columns=["project", "bug_id", "error_msg"])
    failed_tests_df.to_csv(os.path.join(result_dir, "failed_tests.csv"), index=False,encoding='utf-8')


if __name__=='__main__':
    main()

