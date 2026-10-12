# Experimental code

This directory contains the scripts used to evaluate large language models for
Spanish clickbait resolution. The repository compares two formulations of the
task:

- **QA**: answer the question implicit in a clickbait headline using the news
  article as context.
- **Summarization**: generate a headline-conditioned summary that resolves the
  information gap.

Experiments can be run as zero-shot (ZS) inference or supervised fine-tuning
(SFT). The random seed is fixed to `42` and generation is limited to 256 new
tokens.

## Shared evaluation protocol

Both ZS and SFT evaluate the test split with a maximum formatted-prompt length
of 8,192 tokens by default. Set `--max_seq_len` to change this limit. The
limit includes all instructions and chat-template tokens. Llama, Mistral, and
SOLAR use left truncation; all other models retain their tokenizer's default
truncation side. The SFT scripts apply the same limit while preparing training
and validation data.

Generated responses and gold references are trimmed with `strip()` before
ROUGE. Special tokens are removed during decoding. No project-specific
lowercasing, punctuation removal, accent handling, or internal-whitespace
normalization is performed.

## Files

| Path | Purpose |
| --- | --- |
| `zs.py` | Zero-shot inference on the test split, including ROUGE evaluation. |
| `sft_base_version.py` | Supervised fine-tuning for supported base checkpoints. |
| `sft_instructed_version.py` | Supervised fine-tuning for instruction-tuned and chat checkpoints. |
| `deepspeed.json` | DeepSpeed configuration used by the SFT scripts. |

## Input data

The scripts expect a CSV file with the following fields:

| Column | Description |
| --- | --- |
| `Title` | Clickbait headline. |
| `Body` | Full text of the news article. |
| `Response` | Gold extractive answer span. |
| `_split` | Dataset partition: `train`, `val`, or `test`. |

Zero-shot evaluation uses the `test` split. Fine-tuning uses `train` for
training, `val` for validation, and `test` for final ROUGE reporting.

## Supported models

### Zero-shot

| Family | Base checkpoints | Instruction/chat checkpoints |
| --- | --- | --- |
| Gemma 2 | `google/gemma-2-2b` | `google/gemma-2-2b-it` |
| Gemma 3 | `google/gemma-3-1b-pt` | `google/gemma-3-1b-it` |
|  | `google/gemma-3-4b-pt` | `google/gemma-3-4b-it` |
| Llama | `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.1-8B-Instruct` |
|  | `meta-llama/Llama-3.2-1B` | `meta-llama/Llama-3.2-1B-Instruct` |
|  | `meta-llama/Llama-3.2-3B` | `meta-llama/Llama-3.2-3B-Instruct` |
| Mistral | `mistralai/Mistral-7B-v0.3` | `mistralai/Mistral-7B-Instruct-v0.3` |
| SOLAR | `upstage/SOLAR-10.7B-v1.0` | `upstage/SOLAR-10.7B-Instruct-v1.0` |
| Qwen 3 | — | `Qwen/Qwen3-1.7B` |
|  | — | `Qwen/Qwen3-4B` |
|  | — | `Qwen/Qwen3-8B` |
| OpenChat | — | `openchat/openchat-3.5-0106` |
| DeepSeek | — | `deepseek-ai/deepseek-llm-7b-chat` |

Qwen 3, OpenChat, and DeepSeek are intentionally available only through the
instruction/chat workflow; their base variants are not included.

### Supervised fine-tuning

`sft_base_version.py` supports the base Gemma, Llama, Mistral, and SOLAR
checkpoints listed above. `sft_instructed_version.py` supports all the
instruction/chat checkpoints, including Qwen 3, OpenChat, and DeepSeek Chat.

## Requirements

Use Python 3.10 or later. Zero-shot inference and fine-tuning require a CUDA
capable GPU. SFT additionally uses DeepSpeed and the included
`deepspeed.json` configuration, which enables bfloat16 and ZeRO stage 2.

From the repository root, create and activate an isolated environment. With
Conda:

```bash
conda create -n spanish-clickbait python=3.10
conda activate spanish-clickbait
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Alternatively, use Python's built-in `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file pins the direct dependencies used by the scripts:
PyTorch, Transformers, TRL, DeepSpeed, Datasets, Evaluate, and the reporting
libraries. `pip` resolves their transitive dependencies. The pinned TRL
revision is installed from GitHub, so installation requires Git and access to
GitHub.

Several model checkpoints are gated on Hugging Face. Before using one, accept
its terms on the model page and authenticate in the environment:

```bash
huggingface-cli login
```

Verify that PyTorch sees the intended GPUs before starting a multi-GPU run:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

The SFT examples below target GPU `0`. To use additional GPUs, extend the
argument (for example, `--include=localhost:0,1`) and adjust batch sizes to
match the available hardware.

## Running experiments

Run experiments from `code/`. The output directory is created automatically:

```bash
cd code
python zs.py \
  -p ../Spanish_ClickbaitCorpus_2026.csv \
  -m Qwen/Qwen3-4B \
  -a qa \
  --save_path ../results/zs_results
```

For headline-conditioned summarization:

```bash
cd code
python zs.py \
  -p ../Spanish_ClickbaitCorpus_2026.csv \
  -m Qwen/Qwen3-4B \
  -a summ \
  --save_path ../results/zs_results
```

For instruction-tuned or chat models:

```bash
cd code
deepspeed --include=localhost:0 --master_port 62000 \
  sft_instructed_version.py \
  --approach qa \
  --csv_path ../Spanish_ClickbaitCorpus_2026.csv \
  --save_path ../results/sft_results \
  --model_id Qwen/Qwen3-4B \
  --train_bs 2 --eval_bs 2
```

For a supported base checkpoint:

```bash
cd code
deepspeed --include=localhost:0 --master_port 62000 \
  sft_base_version.py \
  --approach summ \
  --csv_path ../Spanish_ClickbaitCorpus_2026.csv \
  --save_path ../results/sft_results \
  --model_id google/gemma-2-2b \
  --train_bs 2 --eval_bs 2
```

Both workflows accept `--save_path` and create it automatically. By default,
ZS results are saved in `../results/zs_results/` and SFT results in
`../results/sft_results/`. Response, prediction, and metric CSV files are
written directly to the selected directory, with the model and approach
included in their filenames. All scripts accept `qa` and `summ` as the only
valid task formulations. Pass `--max_seq_len 8192` explicitly to reproduce
the default prompt limit.
