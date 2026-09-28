import torch
from typing import Dict
from torch.utils.data import Dataset
from src.utils import rank0_print
import transformers
import json
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def load_json_data(data_path):
    """Read JSON format data"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def levenshtein_alignment(seq1, seq2):
    """Use Levenshtein distance for token alignment, return difference positions in seq1"""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + 1)
    
    diff_positions = set()
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i-1] == seq2[j-1]:
            i -= 1
            j -= 1
        else:
            if dp[i-1][j] + 1 == dp[i][j]:
                diff_positions.add(i-1)
                i -= 1
            elif dp[i][j-1] + 1 == dp[i][j]:
                j -= 1
            else:
                diff_positions.add(i-1)
                i -= 1
                j -= 1
    while i > 0:
        diff_positions.add(i-1)
        i -= 1
    
    return diff_positions

def needleman_wunsch_alignment(seq1, seq2, match_score=2, mismatch_score=-1, gap_score=-1):
    """Use Needleman-Wunsch algorithm for token alignment, return difference positions in seq1"""
    m, n = len(seq1), len(seq2)
    score = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        score[i][0] = i * gap_score
    for j in range(1, n + 1):
        score[0][j] = j * gap_score
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                diagonal = score[i-1][j-1] + match_score
            else:
                diagonal = score[i-1][j-1] + mismatch_score
            score[i][j] = max(diagonal, score[i-1][j] + gap_score, score[i][j-1] + gap_score)
    
    diff_positions = set()
    i, j = m, n
    while i > 0 and j > 0:
        current_score = score[i][j]
        if seq1[i-1] == seq2[j-1] and current_score == score[i-1][j-1] + match_score:
            i -= 1
            j -= 1
        elif seq1[i-1] != seq2[j-1] and current_score == score[i-1][j-1] + mismatch_score:
            diff_positions.add(i-1)
            i -= 1
            j -= 1
        elif current_score == score[i-1][j] + gap_score:
            diff_positions.add(i-1)
            i -= 1
        else:
            j -= 1
    while i > 0:
        diff_positions.add(i-1)
        i -= 1
    
    return diff_positions

def preprocess(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        system_message: str = "You are a helpful assistant.",
        alignment_method: str = "lcs"
) -> Dict:

    # Retrieve necessary token IDs
    begin_of_text_id = tokenizer.get_vocab()["<|begin_of_text|>"]
    start_header_id = tokenizer.get_vocab()["<|start_header_id|>"]
    end_header_id = tokenizer.get_vocab()["<|end_header_id|>"]
    eot_id = tokenizer.get_vocab()["<|eot_id|>"]
    nl_tokens = tokenizer('\n\n').input_ids
    _system = tokenizer('system').input_ids
    _user = tokenizer('user').input_ids
    _assistant = tokenizer('assistant').input_ids

    # Process data
    input_ids, targets, output1_positions = [], [], []
    for entry in sources:
        input_id, target = [], []
        
        # Begin with `<|begin_of_text|>` only once
        input_id += [begin_of_text_id]
        target += [IGNORE_TOKEN_ID]

        # Add system message
        system_message_ids = tokenizer(system_message, add_special_tokens=False).input_ids
        system = [start_header_id] + _system + [end_header_id] + nl_tokens + system_message_ids + [eot_id]
        input_id += system
        target += [IGNORE_TOKEN_ID] * len(system)

        # User (instruction) section
        instruction_with_prompt = entry["instruction"] + "\n\nPlease think step by step, and put your final answer within \\boxed{{}}."
        instruction_ids = tokenizer(instruction_with_prompt, add_special_tokens=False).input_ids
        _input_id = [start_header_id] + _user + [end_header_id] + nl_tokens + instruction_ids + [eot_id] 
        _target = [IGNORE_TOKEN_ID] * len(_input_id)
        input_id += _input_id
        target += _target

        has_dual_output = "output1" in entry and "output2" in entry
        
        if has_dual_output:
            output1_ids = tokenizer(entry["output1"], add_special_tokens=False).input_ids
            output2_ids = tokenizer(entry["output2"], add_special_tokens=False).input_ids
        else:
            output1_ids = tokenizer(entry["output"], add_special_tokens=False).input_ids
            

        # Assistant (output1) section
        _input_id = [start_header_id] + _assistant + [end_header_id] + nl_tokens + output1_ids + [eot_id]
        _target = [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(_assistant) + \
                  [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(nl_tokens) + output1_ids + [eot_id]
        
        diff_positions = []
        if has_dual_output:
            assistant_header_len = len([start_header_id] + _assistant + [end_header_id] + nl_tokens)
            base_pos = len(input_id) + assistant_header_len
            
            if alignment_method == "lcs":
                m, n = len(output1_ids), len(output2_ids)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if output1_ids[i-1] == output2_ids[j-1]:
                            dp[i][j] = dp[i-1][j-1] + 1
                        else:
                            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
                i, j = m, n
                output1_diff = set()
                while i > 0 and j > 0:
                    if output1_ids[i-1] == output2_ids[j-1]:
                        i -= 1
                        j -= 1
                    elif dp[i-1][j] >= dp[i][j-1]:
                        output1_diff.add(i-1)
                        i -= 1
                    else:
                        j -= 1
                while i > 0:
                    output1_diff.add(i-1)
                    i -= 1
            elif alignment_method == "levenshtein":
                output1_diff = levenshtein_alignment(output1_ids, output2_ids)
            elif alignment_method == "needleman_wunsch":
                output1_diff = needleman_wunsch_alignment(output1_ids, output2_ids)
            else:
                raise ValueError(f"Unknown alignment method: {alignment_method}")
                
            diff_positions = [base_pos + pos for pos in output1_diff]
            
        input_id += _input_id
        target += _target

        assert input_id[-1] == eot_id, "The final concatenated data did not correctly add <|eot_id|>"

        # Pad to max length
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
        output1_positions.append([pos for pos in diff_positions if pos < max_len])
    
    # Convert to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)


    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        output1_positions=output1_positions,
    )


class SupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_len):
        super(SupervisedDataset, self).__init__()
        sources = raw_data
        data_dict = preprocess(sources, tokenizer, max_len)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        self.output1_positions = data_dict["output1_positions"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
            output1_positions=self.output1_positions[i],
        )


def make_supervised_data_module(tokenizer, data_args, max_len) -> dict:
    """Construct the datasets for training and evaluation."""

    # Load training data
    train_data = load_json_data(data_args.data_path)
    train_dataset = SupervisedDataset(train_data, tokenizer=tokenizer, max_len=max_len)

    # Optionally load evaluation data
    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = SupervisedDataset(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

if __name__ == "__main__":
    make_supervised_data_module()    