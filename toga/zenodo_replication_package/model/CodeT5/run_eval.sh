CUDA_LAUNCH_BLOCKING=1 

input_file=$1
base_dir=`dirname ${input_file}`

python model/CodeT5/codet5_main.py \
--output_dir=model/CodeT5/saved_models \
--model_name=model.bin \
--do_test \
--test_data_file=$input_file \
--encoder_block_size 512 \
--decoder_block_size 256 \
--eval_batch_size 8 \
--num_beams 1 \
--result_output_dir ${base_dir}/CodeT5_preds/ \

