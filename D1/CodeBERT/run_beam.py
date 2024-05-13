import os
import multiprocessing
import subprocess


def run_command(command, output_file):
    with open(output_file, 'w') as f:
        subprocess.call(command, stdout=f, shell=True)
if __name__ == '__main__':
    beams = [2, 3, 4, 5, 10, 30]
    commands = [
            ]
    for i in beams:
        commands.append("python codebert_main.py --output_dir=./saved_models --model_name=model.bin --do_test --test_data_file=../data/fine_tune_data/assert_test.csv --encoder_block_size 512 --decoder_block_size 256 --eval_batch_size 1 --n_gpu 1 --output_name {} --beam_size {}".format(i,i))

    max_concurrent_processes = 5

    pool = multiprocessing.Pool(processes=max_concurrent_processes)




    for idx,command in enumerate(commands):
        output_file = './{}.txt'.format(idx)
        pool.apply_async(run_command, args=(command, output_file))

    pool.close()

    pool.join()