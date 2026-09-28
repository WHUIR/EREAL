"""
https://github.com/allenai/open-instruct
"""
import torch
import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList
import os
from itertools import zip_longest

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_str, tokenizer):
        StoppingCriteria.__init__(self)
        self.current_context = []
        self.tokenizer = tokenizer
        self.keywords_str = keywords_str
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if len(self.current_context) == 0:
            self.current_context = [[] for _ in range(input_ids.shape[0])]

        # 修改为处理每个序列
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            _id = input_ids[i][-1].item()
            self.current_context[i].append(_id)
            current_context = self.tokenizer.decode(self.current_context[i])
            should_be_stopped = False
            for word in self.keywords_str:
                if word in current_context:
                    should_be_stopped = True
                    break
            sequences_should_be_stopped.append(should_be_stopped)
        return all(sequences_should_be_stopped)


class KeyWordsCriteriaTrunc(StoppingCriteria):
    def __init__(self, stop_id_sequences, prompt_length):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            ids = input_ids[i][self.prompt_length:].tolist()
            should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids.shape[0] == 1:
                    _ids = ids[-len(stop_sequence):]
                else:
                    _ids = ids
                for j in range(len(_ids), 0, -len(stop_sequence)):
                    if _ids[max(j - len(stop_sequence), 0): j] == stop_sequence:
                        should_be_stopped = True
                        break
                if should_be_stopped:
                    break
            sequences_should_be_stopped.append(should_be_stopped)
        return all(sequences_should_be_stopped)


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)


@torch.no_grad()
def generate_completions(models, tokenizer, prompts, batch_size=1, stop_id_sequences=None, add_special_tokens=True, disable_tqdm=False, **generation_kwargs):
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    # 计算每个GPU处理的数据量
    n_gpus = len(models)
    chunk_size = len(prompts) // n_gpus
    if len(prompts) % n_gpus != 0:
        chunk_size += 1
    
    prompt_chunks = [prompts[i:i+chunk_size] for i in range(0, len(prompts), chunk_size)]
    
    from concurrent.futures import ThreadPoolExecutor
    
    def process_chunk(args):
        gpu_id, model, prompt_chunk = args
        generations = []
        num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
        
        for i in range(0, len(prompt_chunk), batch_size):
            batch_prompts = prompt_chunk[i:i+batch_size]
            tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=False)
            
            batch_input_ids = tokenized_prompts.input_ids.cuda(gpu_id)
            attention_mask = tokenized_prompts.attention_mask.cuda(gpu_id)

            stop_criteria = KeywordsStoppingCriteria(stop_id_sequences, tokenizer)
            
            try:
                batch_outputs = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=attention_mask, 
                    stopping_criteria=StoppingCriteriaList([stop_criteria]),
                    **generation_kwargs
                )
                
                batch_generations = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(batch_input_ids, batch_outputs)
                ]
                batch_generations = tokenizer.batch_decode(batch_generations, skip_special_tokens=True)

                for idx, prediction in enumerate(batch_generations):
                    for stop_sequence in stop_id_sequences:
                        batch_generations[idx] = prediction.split(stop_sequence)[0]

                generations.extend(batch_generations)

            except Exception as e:
                print(f"Error on GPU {gpu_id} for batch {i}: {str(e)}")
                continue

            if not disable_tqdm:
                progress.update(len(batch_prompts)//num_return_sequences)
                
        return generations

    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        all_generations = list(executor.map(process_chunk, 
                                         [(gpu_id, model, chunk) 
                                          for gpu_id, (model, chunk) 
                                          in enumerate(zip(models, prompt_chunks))]))
    
    final_generations = []
    for gens in zip_longest(*all_generations, fillvalue=None):
        final_generations.extend([g for g in gens if g is not None])
        
    return final_generations[:len(prompts)]


def load_hf_lm_and_tokenizer(
        model_name_or_path, 
        tokenizer_name_or_path=None, 
        device_map="auto", 
        load_in_8bit=False, 
        load_in_half=True,
        gptq_model=False,
        use_fast_tokenizer=False,
        padding_side="left",
        use_safetensors=False,
    ):
    import torch 
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer, padding_side=padding_side, trust_remote_code=True)


    # set pad token to eos token if pad token is not set
    if tokenizer.pad_token is None:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            raise ValueError("You are using a new tokenizer without a pad token."
                            "This is not supported by this script.")

    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.unk_token
    #     tokenizer.pad_token_id = tokenizer.unk_token_id

    if gptq_model:
        from auto_gptq import AutoGPTQForCausalLM
        model_wrapper = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path, device="cuda:0", use_triton=True
        )
        model = model_wrapper.model
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map=device_map, 
            load_in_8bit=True
        )
    else:
        available_gpus = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        models = []
        for gpu_id in available_gpus:
            torch.cuda.set_device(gpu_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map={"": gpu_id},
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).eval()
            models.append(model)
            print(f"Model loaded on GPU {gpu_id}")
            
    return models, tokenizer


def _test_generate_completions():
    model_name_or_path = "../models/codellama_7b/v1-16k"
    llm, tokenizer = load_hf_lm_and_tokenizer(
                        model_name_or_path=model_name_or_path, 
                        load_in_half=True,
                        use_fast_tokenizer=True,
                        use_safetensors=True,
                    )
    # some math word problems
    prompts = [
        "---\n1+1=2\n---2+2=4\n---3+3=6\n---4+4=8\n---5+5=10\n---6+6=",
        "---\n1+1=2\n---12+12=24\n---3+3=6\n---12345+12345=",
        # "A train leaves Chicago at 7am and travels at 60mph. Another train leaves Chicago at 9am and travels at 80mph. When will the second train overtake the first?",
        # "The sum of two numbers is 10. The difference of the same two numbers is 4. What are the two numbers?",
    ]

    stop_sequences = ["\n\n\n", "---"]
    # Because many tokenizers will treat the word after space differently from the original word alone, 
    # to be consistent, we add a space before tokenization and remove it after tokenization.
    # stop_id_sequences = [tokenizer.encode(" " + x, add_special_tokens=False)[1:] for x in stop_sequences]
    outputs = generate_completions(
            model=llm,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=128,
            batch_size=16,
            # stop_id_sequences=stop_id_sequences,
            stop_id_sequences=stop_sequences,
    )
    print(outputs)

if __name__ == "__main__":
    _test_generate_completions()