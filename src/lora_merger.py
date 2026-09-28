from typing import Union, List
from transformers import PreTrainedModel
from peft import PeftModel
import torch
from transformers import AutoTokenizer

class LoRAMerger:
    def __init__(
        self,
        base_model_path: str,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the LoRA merger
        
        Args:
            base_model_path: The path to the base model
            device: The device type "auto", "cpu", "cuda", "cuda:0" etc.
            torch_dtype: The model weight type
        """
        self.base_model_path = base_model_path
        # self.device = device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.torch_dtype = torch_dtype
        
    def load_base_model(self) -> PreTrainedModel:
        """Load the base model"""
        from transformers import AutoModelForCausalLM
        
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device
        )
        return model

    def merge_lora_weights(
        self,
        model: PreTrainedModel,
        adapter_paths: Union[str, List[str]],
        save_path: str,
        **kwargs
    ) -> PreTrainedModel:
        """
        Merge the LoRA weights
        
        Args:
            model: The base model
            adapter_paths: The path to the LoRA adapter, can be a single path or a list of paths
            save_path: The path to save the merged model
            **kwargs: Other parameters, passed to from_pretrained
        
        Returns:
            The merged model
        """
        if isinstance(adapter_paths, str):
            adapter_paths = [adapter_paths]
            
        # Merge multiple adapters one by one
        for adapter in adapter_paths:
            print(f"Merging adapter: {adapter}")
            model = PeftModel.from_pretrained(
                model,
                adapter,
                **kwargs
            )
            model = model.merge_and_unload()
            
        # Save the merged model
        model.save_pretrained(save_path)
        return model

    def merge_and_save(self, adapter_paths, save_path, **kwargs):
        model = self.load_base_model()
        self.merge_lora_weights(model, adapter_paths, save_path, **kwargs)
        
        # Save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        tokenizer.save_pretrained(save_path)
        
        print(f"Merged model and tokenizer saved to: {save_path}")
