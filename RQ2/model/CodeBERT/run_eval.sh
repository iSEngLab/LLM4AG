CUDA_LAUNCH_BLOCKING=1 

input_file=$1
base_dir=`dirname ${input_file}`

python model/CodeBERT/codebert_main.py \
--output_dir=model/CodeBERT/saved_models \
--model_name=model.bin \
--test_data_file=$input_file \
--encoder_block_size 512 \
--decoder_block_size 256 \
--eval_batch_size 8 \
--beam_size 1 \
--result_output_dir ${base_dir}/CodeBERT_preds/ \

