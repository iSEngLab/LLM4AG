CUDA_LAUNCH_BLOCKING=1 

input_file=$1
base_dir=`dirname ${input_file}`

python model/CodeGPT/run.py \
         --do_test \
         --model_type gpt2 \
         --output_file_name new \
         --load_model_path model/CodeGPT/24-Jan-2023/checkpoint-best-ppl/new.bin \
         --model_name_or_path microsoft/CodeGPT-small-java-adaptedGPT2 \
         --test_filename $input_file  \
         --output_dir ./24-Jan-2023/ \
         --max_source_length 512 \
         --max_target_length 512 \
         --beam_size 1 \
         --train_batch_size 8 \
         --eval_batch_size 8 \
         --result_output_dir ${base_dir}/CodeGPT_preds/ \
         --learning_rate 5e-5 \
         --num_train_epochs 30 \
