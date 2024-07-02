CUDA_IDS=$1
OUTPUT_DIR=$2
DATA_PATH=$3

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_IDS} accelerate launch llama_pred.py \
            --model_name_or_path ${OUTPUT_DIR} \
            --model_max_length 1024 \
            --data_path ${DATA_PATH} \
            --test_filename test.jsonl \
            --output_dir ${OUTPUT_DIR} \
            --max_new_tokens 256 \
            --max_length 1024 \
            --request_num 1 \
            --num_beams 10