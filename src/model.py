import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import is_deepspeed_zero3_enabled
import transformers
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class CustomLlamaModel(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        output2_positions=None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if self.training and output2_positions is not None and labels is not None:
            loss_weights = torch.ones_like(labels, dtype=torch.float)
            
            for batch_idx, pos_list in enumerate(output2_positions):
                if len(pos_list) > 0:
                    valid_tokens = (labels[batch_idx] != IGNORE_TOKEN_ID).sum().item()
                    diff_tokens = len(pos_list)
                    weight = 1 + self.config.weight_value * (diff_tokens / valid_tokens)
                    # weight = 1 + self.config.weight_value * (1 - diff_tokens / valid_tokens)
                    # weight = 1 + self.config.weight_value
                    # print(f"weight: {weight:.4f}")

                    for pos in pos_list:
                        loss_weights[batch_idx, pos] = weight
            
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_loss_weights = loss_weights[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                            shift_labels.view(-1))
            
            weighted_losses = losses * shift_loss_weights.view(-1)
            valid_mask = (shift_labels.view(-1) != IGNORE_TOKEN_ID).float()
            outputs.loss = (weighted_losses * valid_mask).sum() / (valid_mask.sum() + 1e-6)
            
        return outputs

def load_model_and_tokenizer(model_args, training_args, lora_args):
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    config.use_improved_loss = training_args.use_improved_loss
    config.weight_value = training_args.weight_value 

    if training_args.use_improved_loss:
        model = CustomLlamaModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            device_map=None,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            device_map=None,
            trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Apply LoRA configuration if specified
    if training_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=lora_args.lora_target_modules,
            task_type="CAUSAL_LM",
        )


        model = get_peft_model(model, lora_config)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    return model, tokenizer