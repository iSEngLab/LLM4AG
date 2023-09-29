import os
if __name__ =="__main__":
    os.system("python codet5_main.py  --do_test --test_data_file=../data/fine_tune_data/assert_test.csv --encoder_block_size 512 --decoder_block_size 256 --num_beams 3 --eval_batch_size 1 --n_gpu 1")
