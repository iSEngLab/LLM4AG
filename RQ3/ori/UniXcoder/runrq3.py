import os
if __name__ =="__main__":
    os.system("python unixcoder_main.py --output_dir=./saved_models --model_name=kt.bin --do_train --train_data_file=../data/fine_tune_data/kotlin_data_10000_train.csv --eval_data_file=../data/fine_tune_data/kotlin_data_10000_val.csv --epochs 75 --encoder_block_size 512 --decoder_block_size 256 --train_batch_size 8 --eval_batch_size 8 --learning_rate 2e-5 --max_grad_norm 1.0 --n_gpu 1 --evaluate_during_training --seed 123456  2>&1 | tee train.log")
    os.system("python unixcoder_main.py --output_name kt2java --output_dir=./saved_models --model_name=kt.bin --do_test --test_data_file=../data/fine_tune_data/assert_test_new.csv --encoder_block_size 512 --decoder_block_size 256 --eval_batch_size 1 --beam_size 1 --n_gpu 1")
    os.system("python unixcoder_main.py --output_dir=./saved_models --model_name=kt2java.bin  --load_model_from_checkpoint --checkpoint_model_name kt.bin --do_train --train_data_file=../data/fine_tune_data/assert_train_new.csv --eval_data_file=../data/fine_tune_data/assert_val_new.csv --test_data_file=../data/fine_tune_data/assert_test_new.csv --epochs 75 --encoder_block_size 512 --decoder_block_size 256 --train_batch_size 8 --eval_batch_size 8 --learning_rate 2e-5 --max_grad_norm 1.0 --n_gpu 1 --evaluate_during_training --seed 123456  2>&1 | tee train.log")
    os.system("python unixcoder_main.py --output_name kt2javaaftertrained --output_dir=./saved_models --model_name=kt2java.bin --do_test --test_data_file=../data/fine_tune_data/assert_test_new.csv --encoder_block_size 512 --decoder_block_size 256 --eval_batch_size 1 --beam_size 1 --n_gpu 1")

