OUTPUT_DIR="/path/to/EREAL/output/model_name_or_path"
DATA_PATH="/path/to/EREAL/data/train.json"

NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
--nproc_per_node 4 \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 6601 \
../main.py \
--model_name_or_path "/path/to/models/Meta-Llama-3.1-8B-Instruct" \
--data_path "$DATA_PATH" \
--bf16 True \
--output_dir "$OUTPUT_DIR" \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--save_strategy "epoch" \
--save_steps 1 \
--learning_rate 1e-5 \
--weight_decay 0.1 \
--adam_beta2 0.95 \
--warmup_ratio 0.01 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--report_to "none" \
--model_max_length 2560 \
--gradient_checkpointing True \
--use_lora \
--use_flash_attention \
--use_improved_loss True \
--weight_value 2.0 \
--deepspeed "/path/to/EREAL/config/ds_config_zero0.json"
