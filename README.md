# EREAL: Enhancing Mathematical Reasoning through Error-Aware Learning in Large Language Models

[![Paper](https://img.shields.io/badge/Paper-IPM__105197-blue)](https://doi.org/10.1016/j.ipm.2026.105197)

Official implementation of **EREAL** (Enhancing Reasoning through Error-Aware Learning), a novel framework that enables LLMs to learn from token-level mistakes for improved mathematical reasoning.

## 📖 Overview

EREAL addresses the limitations of existing error-correction methods by introducing:
- **Minimal-edit Correction**: Revises only erroneous segments while preserving valid reasoning steps
- **Dynamic Error-Aware Loss**: Adaptively weights tokens based on error proportion during training
- **Fine-grained Supervision**: Enables token-level learning from mistakes

> 🔬 **Key Insight**: Most model errors require only localized corrections, making coarse-grained rewriting inefficient and disruptive to valid reasoning.

## 🚀 Quick Start

### Installation
```bash
cd EREAL
pip install -r requirements.txt
```
### Training
To train a model using EREAL:
```bash
# Edit the paths in finetune.sh first
# - Set OUTPUT_DIR to your desired output directory
# - Set DATA_PATH to your training data
# - Set model_name_or_path to your base model path
bash ./script/finetune.sh
```

### Evaluation
To evaluate your fine-tuned model:
```bash
# First merge the LoRA weights with the base model
# Edit the paths in evaluate.sh:
# - Set base-model to your base model path
# - Set lora-path to your checkpoint directory
# - Set save-path to where you want to save the merged model

# Then run the evaluation script
bash ./script/evaluate.sh
```



## 📊 Supported Models & Benchmarks

### Tested Models
- **LLaMA-3.1-8B-Instruct**
- **DeepSeek-Math-7B-Instruct** 
- **Mistral-7B-Instruct-v0.3**

### Evaluation Benchmarks
| Category | Dataset | Samples |
|----------|---------|---------|
| **In-Distribution** | GSM8K | 1,319 |
| | MATH | 5,000 |
| **Out-of-Distribution** | ASDiv | 2,215 |
| | MAWPS | 2,065 |
| | SVAMP | 1,000 |
| | TabMWP | 1,000 |
| | CARP_EN | 976 |

## 📚 Citation

If you find this code useful, please cite our paper:

```bibtex
@article{YU2027105197,
  title   = {EREAL: Enhancing Mathematical Reasoning through Error-Aware Learning in Large Language Models},
  journal = {Information Processing & Management},
  volume  = {64},
  number  = {2, Part B},
  pages   = {105197},
  year    = {2027},
  issn    = {0306-4573},
  doi     = {10.1016/j.ipm.2026.105197},
  url     = {https://www.sciencedirect.com/science/article/pii/S030645732600587X},
  author  = {Yuqing Yu and Zihao Li and Lixin Zou and Chenliang Li and Qian Wang}
}
```

