import os
from transformers import Trainer
from src.utils import safe_save_model_for_hf_trainer


class CustomTrainer(Trainer):
    def __init__(self, lora_bias="none", **kwargs):
        super().__init__(**kwargs)
        self.lora_bias = lora_bias
        

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        safe_save_model_for_hf_trainer(
            trainer=self,
            output_dir=output_dir,
            bias=self.lora_bias,
        )