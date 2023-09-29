import os
if __name__ == '__main__':
    os.system('python run.py \
        --do_train \
        --do_eval \
        --model_type gpt2 \
        --output_model_name new.bin \
        --model_name_or_path microsoft/CodeGPT-small-java-adaptedGPT2 \
        --train_filename ../data/fine_tune_data/assert_train_new.csv \
        --dev_filename ../data/fine_tune_data/assert_val_new.csv  \
        --test_filename ../data/fine_tune_data/assert_test_new.csv  \
        --output_dir ./24-Jan-2023/ \
        --max_source_length 512 \
        --max_target_length 512 \
        --beam_size 1 \
        --train_batch_size 8 \
        --eval_batch_size 8 \
        --learning_rate 5e-5 \
        --num_train_epochs 30 \
        2>&1 | tee ./24-Jan-2023/train.log')
    os.system('python run.py \
         --do_test \
         --model_type gpt2 \
         --output_file_name new \
         --load_model_path ./24-Jan-2023/checkpoint-best-ppl/new.bin \
         --model_name_or_path microsoft/CodeGPT-small-java-adaptedGPT2 \
         --test_filename ../data/fine_tune_data/assert_test_new.csv  \
         --output_dir ./24-Jan-2023/ \
         --max_source_length 512 \
         --max_target_length 512 \
         --beam_size 1 \
         --train_batch_size 8 \
         --eval_batch_size 8 \
         --learning_rate 5e-5 \
         --num_train_epochs 30 \
         2>&1 | tee ./24-Jan-2023/eval-23-Aug.log')
    os.system('python run.py \
        --do_train \
        --do_eval \
        --model_type gpt2 \
        --output_model_name old.bin \
        --model_name_or_path microsoft/CodeGPT-small-java-adaptedGPT2 \
        --train_filename ../data/fine_tune_data/assert_train_old.csv \
        --dev_filename ../data/fine_tune_data/assert_val_old.csv  \
        --output_dir ./24-Jan-2023/ \
        --max_source_length 512 \
        --max_target_length 512 \
        --beam_size 1 \
        --train_batch_size 8 \
        --eval_batch_size 8 \
        --learning_rate 5e-5 \
        --num_train_epochs 30 \
        2>&1 | tee ./24-Jan-2023/train.log')
    os.system('python run.py \
         --do_test \
         --model_type gpt2 \
         --output_file_name old \
         --load_model_path ./24-Jan-2023/checkpoint-best-ppl/old.bin \
         --model_name_or_path microsoft/CodeGPT-small-java-adaptedGPT2 \
         --test_filename ../data/fine_tune_data/assert_test_old.csv  \
         --output_dir ./24-Jan-2023/ \
         --max_source_length 512 \
         --max_target_length 512 \
         --beam_size 1 \
         --train_batch_size 8 \
         --eval_batch_size 8 \
         --learning_rate 5e-5 \
         --num_train_epochs 30 \
         2>&1 | tee ./24-Jan-2023/eval-23-Aug.log')