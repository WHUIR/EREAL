from dataclasses import dataclass, field
from typing import List, Optional
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the evaluation data."})
    alignment_method: str = field(
        default="lcs",
        metadata={
            "help": "Token alignment method for comparing output1 and output2. Options: 'lcs', 'levenshtein', 'needleman_wunsch'"
        }
    )

@dataclass
class CustomTrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    use_flash_attention: bool = False
    use_improved_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use improved loss calculation method"}
    )
    weight_value: float = field(
        default=5.0,
        metadata={"help": "Weight coefficient for improved loss calculation"}
    )


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    lora_target_modules: List[str] = field(
        default_factory=lambda: ['gate_proj', 'o_proj', 'k_proj', 'q_proj', 'up_proj', 'down_proj', 'v_proj']
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"