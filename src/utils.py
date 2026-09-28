import os
import json
import torch
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed import zero

local_rank = None

def rank0_print(*args):
    """Only print for rank0 process, for distributed training."""
    if local_rank == 0:
        print(*args)

def set_local_rank(rank):
    """Set the value of local_rank."""
    global local_rank
    local_rank = rank

def maybe_zero_3(param):
    """Handle the ZeRO partition parameters."""
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    """Collect LoRA weights and optional bias from parameter names."""
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for bias_name in lora_bias_names:
            if bias_name in maybe_lora_bias:
                to_return[bias_name] = maybe_lora_bias[bias_name]
    else:
        raise NotImplementedError(f"Bias option {bias} is not implemented!")
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer, output_dir, bias="none"):
    """Safely saves HuggingFace model weights with optional LoRA and bias."""
    if is_deepspeed_zero3_enabled(trainer.args):
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def is_deepspeed_zero3_enabled(training_args):
    """Checks if DeepSpeed ZeRO stage 3 is enabled."""
    if not training_args.deepspeed:
        return False
    with open(training_args.deepspeed, "r") as f:
        ds_config = json.load(f)
    return ds_config.get("zero_optimization", {}).get("stage", 0) == 3
