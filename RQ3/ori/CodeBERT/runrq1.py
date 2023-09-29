import os
if __name__ =="__main__":
    os.system("python codebert_main.py --output_dir=./saved_models --model_name=new.bin --do_train --train_data_file=../data/fine_tune_data/assert_train_new.csv --eval_data_file=../data/fine_tune_data/assert_val_new.csv --test_data_file=../data/fine_tune_data/assert_test_new.csv --epochs 75 --encoder_block_size 512 --decoder_block_size 256 --train_batch_size 8 --eval_batch_size 8 --learning_rate 2e-5 --max_grad_norm 1.0 --n_gpu 1 --evaluate_during_training --seed 123456  2>&1 | tee train.log")
    os.system("python codebert_main.py --output_name new --output_dir=./saved_models --model_name=new.bin --do_test --test_data_file=../data/fine_tune_data/assert_test_new.csv --encoder_block_size 512 --decoder_block_size 256 --beam_size 1 --eval_batch_size 1 --n_gpu 1")
    os.system("python codebert_main.py --output_dir=./saved_models --model_name=old.bin --do_train --train_data_file=../data/fine_tune_data/assert_train_old.csv --eval_data_file=../data/fine_tune_data/assert_val_old.csv --test_data_file=../data/fine_tune_data/assert_test_old.csv --epochs 75 --encoder_block_size 512 --decoder_block_size 256 --train_batch_size 8 --eval_batch_size 8 --learning_rate 2e-5 --max_grad_norm 1.0 --n_gpu 1 --evaluate_during_training --seed 123456  2>&1 | tee train.log")
    os.system("python codebert_main.py --output_name old --output_dir=./saved_models --model_name=old.bin --do_test --test_data_file=../data/fine_tune_data/assert_test_old.csv --encoder_block_size 512 --decoder_block_size 256 --beam_size 1 --eval_batch_size 1 --n_gpu 1")
