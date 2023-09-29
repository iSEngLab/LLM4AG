import os
import multiprocessing
import subprocess


def run_command(command, output_file):
    with open(output_file, 'w') as f:
        subprocess.call(command, stdout=f, shell=True)
if __name__ == '__main__':
    beams = [2, 3, 4, 5, 10, 30,50]
    commands = [
            ]
    for i in beams:
        commands.append("python run.py \
         --do_test \
         --model_type gpt2 \
         --output_file_name {} \
         --load_model_path ./24-Jan-2023/checkpoint-best-ppl/new.bin \
         --model_name_or_path microsoft/CodeGPT-small-java-adaptedGPT2 \
         --test_filename ../data/fine_tune_data/assert_test_new.csv  \
         --output_dir ./24-Jan-2023/ \
         --max_source_length 512 \
         --max_target_length 512 \
         --beam_size {} \
         --train_batch_size 8 \
         --eval_batch_size 8 \
         --learning_rate 5e-5 \
         --num_train_epochs 30 \
         2>&1 | tee ./24-Jan-2023/eval-23-Aug.log".format(i,i))

    max_concurrent_processes = 5

    pool = multiprocessing.Pool(processes=max_concurrent_processes)




    for idx,command in enumerate(commands):
        output_file = './{}.txt'.format(idx)
        pool.apply_async(run_command, args=(command, output_file))

    pool.close()

    pool.join()