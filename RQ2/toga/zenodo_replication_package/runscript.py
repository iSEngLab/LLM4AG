import os
import multiprocessing
import subprocess


def run_command(command, output_file):
    with open(output_file, 'w') as f:
        subprocess.call(command, stdout=f, shell=True)
if __name__ == '__main__':

    commands = [
            ]
    for i in (6,11):
        commands.append('rm -rf /data/swf/zenodo_replication_package_new/data/evosuite_buggy_tests/{}/naive_generated'.format(i))

    max_concurrent_processes = 5

    pool = multiprocessing.Pool(processes=max_concurrent_processes)




    for idx,command in enumerate(commands):
        output_file = './{}.txt'.format(idx)
        pool.apply_async(run_command, args=(command, output_file))

    pool.close()

    pool.join()