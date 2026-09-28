CUDA_VISIBLE_DEVICES=0,1,2,3 python /path/to/EREAL/merge.py \
    --base-model /path/to/models/Meta-Llama-3.1-8B-Instruct \
    --lora-path /path/to/EREAL/output/model_name_or_path/checkpoint \
    --save-path /path/to/EREAL/output/model_name_or_path/merged/epoch3 \
    --device auto \


###Meta-Llama-3.1-8B-Instruct
set -ex

PROMPT_TYPE=llama3-1

MODEL_NAME_OR_PATH=/path/to/EREAL/output/model_name_or_path/merged/epoch3
OUTPUT_DIR=/path/to/EREAL/evaluation/results

SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
DATA_NAME="gsm8k,math,svamp,asdiv,mawps,carp_en,tabmwp"
TOKENIZERS_PARALLELISM=false \
export CUDA_VISIBLE_DEVICES=0,1,2,3
# cd ../
python3 -u /path/to/EREAL/eval/Qwen2.5-Math/evaluation/math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 5 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \

