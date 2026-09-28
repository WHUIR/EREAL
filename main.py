import os
import time
from datetime import datetime
from transformers import HfArgumentParser
from src.arguments import ModelArguments, DataArguments, CustomTrainingArguments, LoraArguments
from src.model import load_model_and_tokenizer
from src.dataset import make_supervised_data_module
from src.trainer import CustomTrainer


def train():
    # Add start time recording
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    
    # Ensure the output directory exists
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Create a simple time log file
    with open(os.path.join(training_args.output_dir, "training_time.log"), "a") as f:
        f.write(f"Training start time: {start_datetime}\n")

    model, tokenizer = load_model_and_tokenizer(model_args, training_args, lora_args)

    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length,
    )

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        lora_bias=lora_args.lora_bias,
        **data_module,
    )

    trainer.train()
    trainer.save_model(output_dir=training_args.output_dir)

    # Record the end time
    end_time = time.time()
    duration = end_time - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Write the training time to the log file
    with open(os.path.join(training_args.output_dir, "training_time.log"), "a") as f:
        f.write(f"Training end time: {end_datetime}\n")
        f.write(f"Training total duration: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds\n")


if __name__ == "__main__":
    train()