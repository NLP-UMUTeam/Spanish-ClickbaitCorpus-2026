# Spanish ClickbaitCorpus 2026

## Question Answering vs. Summarization for Clickbait Resolution in Spanish

**GitHub Repository:** [NLP-UMUTeam/Spanish-ClickbaitCorpus-2026](https://github.com/NLP-UMUTeam/Spanish-ClickbaitCorpus-2026)

Spanish ClickbaitCorpus 2026 is a Spanish-language resource for clickbait
resolution. Each example pairs a clickbait-style news headline with an
extractive answer span from its article body that resolves the headline's
informational gap.

This repository contains the corpus, the prompts and code used to compare two
ways of modelling the task with open-weight language models:

- **Question answering (QA):** interpret the headline as an implicit question
  and answer it from the article.
- **Headline-conditioned summarization:** generate a concise summary that
  resolves the headline's curiosity gap.

The experiments cover zero-shot (ZS) inference and supervised fine-tuning
(SFT), and report lexical and semantic metrics.

## Authors

- **Tomás Bernal-Beltrán** — University of Murcia — [Google Scholar](https://scholar.google.com/citations?user=0bTUxQEAAAAJ) · [ORCID](https://orcid.org/0009-0006-6971-1435)
- **Ronghao Pan** — University of Murcia — [Google Scholar](https://scholar.google.com/citations?user=80lntLMAAAAJ) · [ORCID](https://orcid.org/0009-0008-7317-7145)
- **José Antonio García-Díaz** — University of Murcia — [Google Scholar](https://scholar.google.com/citations?user=ek7NIYUAAAAJ) · [ORCID](https://orcid.org/0000-0002-3651-2660)
- **Rafael Valencia-García** — University of Murcia — [Google Scholar](https://scholar.google.com/citations?user=GLpBPNMAAAAJ) · [ORCID](https://orcid.org/0000-0003-2457-1791)

*Departamento de Informática y Sistemas, Universidad de Murcia, Campus de
Espinardo, 30100 Murcia, Spain.*

## Citation

> **Publication note:** the DOI and citation information will be added
> after acceptance of the associated paper.

## Acknowledgments

This work is part of the research projects LaTe4PoliticES
(PID2022-138099OB-I00) and EmpaTIC-ES (PID2025-169617OB-I00), funded by
MICIU/AEI/10.13039/501100011033 and the European Regional Development Fund
(ERDF/EU) — a way of making Europe. Tomás Bernal-Beltrán is supported by the
University of Murcia through its predoctoral programme.

## Abstract
This paper presents a systematic comparison of Question Answering (QA) and summarization paradigms for clickbait resolution in Spanish. We introduce Spanish ClickbaitCorpus 2026, a new dataset comprising 1,148 Spanish news articles with clickbait-style headlines, each paired with an extractive answer span designed to resolve the informational gap induced by the headline. Using this corpus, we evaluate several open-source Large Language Models (LLMs) under both Zero-Shot (ZS) and Supervised Fine-Tuning (SFT) settings. Performance is assessed using complementary lexical and semantic metrics, including Recall-Oriented Understudy for Gisting Evaluation (ROUGE-1, ROUGE-2, ROUGE-L and ROUGE-Lsum), the BERT-based semantic similarity metric BERTScore F1, and a Hybrid indicator that combines structural and contextual alignment. Statistical significance is evaluated through paired bootstrap resampling. Across models and experimental conditions, QA consistently outperforms summarization. The advantage is particularly pronounced in the ZS setting and remains statistically reliable after SFT, although the performance gap narrows with task-specific adaptation. These findings suggest that framing clickbait resolution as an explicit QA problem provides a stronger structural inductive bias, promoting span-level grounding and more stable resolution of the informational gap. Future work will focus on developing Spanish-language LLMs specifically tuned for clickbait resolution and expanding the corpus with additional multimodal sources and annotations to strengthen coverage and representativeness.

## Overview

The experimental workflow is:

```text
Spanish news article + clickbait headline
                  |
                  v
      Extractive answer-span annotation
                  |
                  v
     QA or headline-conditioned summarization
                  |
                  v
      Zero-shot inference or supervised fine-tuning
                  |
                  v
      ROUGE, BERTScore, hybrid metric, and significance testing
```

Across the evaluated configurations, QA generally achieves higher ROUGE-L and Hybrid scores than headline-conditioned summarization, particularly under zero-shot conditions. However, the relative performance of both formulations varies across model families, especially after supervised fine-tuning.

## Repository structure

```text
.
├── README.md
├── Spanish_ClickbaitCorpus_2026.csv  # Corpus and split assignments
├── requirements.txt                  # Direct Python dependencies
├── code/                             # ZS and SFT experiment scripts
├── prompts/                          # Prompt templates used by the scripts
├── results/                          # Generated predictions and summaries
└── stats/                            # Tables, figures, and metric utilities
```

See `code/README.md` for installation instructions, supported
models, and the exact commands to run the experiments.


## Experimental methodology

The experiments compare two task formulations, QA and headline-conditioned summarization, under ZS and SFT conditions.

### Task formulations

Both formulations use the clickbait headline and the corresponding news article body as input.

- **QA:** The model is instructed to identify the information that directly answers the implicit question raised by the headline.
- **Summarization:** The model is instructed to generate a concise, headline-conditioned summary that resolves the informational gap.

Both formulations are evaluated against the same gold response provided in the `Response` column of the dataset.

### Evaluated models

The experiments consider several open-weight, decoder-based LLM families with different parameter scales:

- **Gemma:** Gemma 2 2B and Gemma 3 1B/4B.
- **LLaMA:** LLaMA 3.1 8B and LLaMA 3.2 1B/3B.
- **Qwen3:** Qwen3 1.7B, 4B and 8B.
- **Mistral:** Mistral 7B v0.3.
- **OpenChat:** OpenChat 3.5 (0106).
- **DeepSeek:** DeepSeek LLM 7B.
- **SOLAR:** SOLAR 10.7B.

Both base and instruction-tuned checkpoints were evaluated where available in the experimental selection. For Qwen3, OpenChat 3.5 and DeepSeek LLM, only instruction-tuned checkpoints were included.

Each selected checkpoint was evaluated under both QA and summarization formulations in ZS and SFT settings.

The exact model identifiers and model-specific execution instructions are provided in `code/README.md`.

### Zero-shot inference

In the ZS setting, pretrained checkpoints are evaluated without additional task-specific parameter updates.

The experiments use English-language instructions and require responses in Spanish. Separate prompt templates are provided for QA and summarization.

Generation uses greedy decoding (`do_sample=False`), with a maximum of 256 new tokens. Input sequences are limited to 8,192 tokens.

### Supervised fine-tuning

In the SFT setting, each selected checkpoint is independently fine-tuned for QA and summarization using the training partition of Spanish ClickbaitCorpus 2026. Both formulations use the same gold response from the Response column as the target output, while differing in their task-specific instructions. This design allows the effect of task formulation to be examined while maintaining a common reference target.

Fine-tuning is performed using Hugging Face TRL's `SFTTrainer`, with the following configuration:

| Parameter | Value |
| --- | --- |
| Fine-tuning strategy | Full-parameter supervised fine-tuning |
| Training epochs | 5 |
| Per-device training batch size | 2 |
| Gradient accumulation steps | 2 |
| Maximum sequence length | 8,192 tokens |
| Optimizer | AdamW |
| Learning rate | 5e-6 |
| Adam beta1 | 0.9 |
| Adam beta2 | 0.95 |
| Adam epsilon | 1e-12 |
| Weight decay | 0 |
| Learning rate scheduler | Cosine |
| Warm-up ratio | 0.1 |
| Random seed | 42 |
| Mixed precision | BF16 |
| Distributed optimization | DeepSpeed ZeRO Stage 2 |
| Optimizer offloading | CPU |
| Gradient checkpointing | Enabled |
| Validation frequency | Every epoch |
| Checkpoint selection | Final epoch |
| Maximum generated tokens | 256 |
| Decoding strategy | Greedy (`do_sample=False`) |

No early stopping or best-checkpoint selection is applied. The final checkpoint obtained after the configured training epochs is used for evaluation.

### Evaluation protocol

The dataset contains predefined training, validation and test partitions, identified by the `_split` column.

- The training partition is used for supervised fine-tuning.
- The validation partition is used to monitor validation loss during training.
- The test partition is used for the final evaluation of both ZS and SFT configurations.

Model predictions are evaluated using ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum, BERTScore F1 and the Hybrid indicator.

The Hybrid indicator is calculated as the arithmetic mean of ROUGE-L and BERTScore F1, with both metrics expressed on the same numerical scale.

## Dataset

The corpus is distributed as `Spanish_ClickbaitCorpus_2026.csv`. It contains
1,148 news articles split into train, validation, and test partitions.

| Split | Articles | Average headline words | Average article words | Average answer words |
| --- | ---: | ---: | ---: | ---: |
| Train | 918 | 16 | 444 | 12 |
| Validation | 115 | 16 | 446 | 12 |
| Test | 115 | 17 | 418 | 12 |

### Fields

| Field | Description |
| --- | --- |
| `URL` | URL of the source article. |
| `Newspaper` | Publishing outlet. |
| `Title` | Clickbait-style headline. |
| `Author` | Byline, when available. |
| `Body` | Article body. |
| `Response` | Extractive answer span resolving the headline. |
| `Class` | Article topic. |
| `Region` | Outlet region: `Spain` or `Latin America`. |
| `_split` | Experimental partition: `train`, `val`, or `test`. |

### Category distribution

The following distribution uses English labels for presentation.

| Class | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Current affairs | 283 | 46 | 30 | 359 |
| Sports | 105 | 10 | 16 | 131 |
| Celebrities | 87 | 4 | 8 | 99 |
| Health | 83 | 9 | 10 | 102 |
| Gastronomy | 57 | 9 | 7 | 73 |
| Cinema | 55 | 5 | 8 | 68 |
| Tourism | 44 | 6 | 9 | 59 |
| Economy | 44 | 3 | 3 | 50 |
| Technology | 28 | 6 | 6 | 40 |
| Weather | 27 | 2 | 3 | 32 |
| Legislation | 23 | 3 | 4 | 30 |
| Television | 19 | 1 | 1 | 21 |
| Security | 17 | 1 | 3 | 21 |
| Science | 12 | 4 | 0 | 16 |
| Language | 8 | 3 | 2 | 13 |
| Automotive | 2 | 0 | 0 | 2 |
| Video games | 9 | 1 | 1 | 11 |
| Music | 6 | 1 | 3 | 10 |
| Beauty | 8 | 0 | 1 | 9 |
| Politics | 1 | 1 | 0 | 2 |

## Evaluation

Predictions are assessed with ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum,
BERTScore F1, and the hybrid metric proposed in the associated work. The
reported QA-versus-summarization comparisons use paired bootstrap resampling
with 1,000 resamples for ROUGE-L and the hybrid metric.

Utilities for assembling result tables and figures are in `stats/`:

| Script | Purpose |
| --- | --- |
| `stats/calculate_metrics_zs.py` | Generates the zero-shot LaTeX results table from saved outputs. |
| `stats/calculate_metrics_sft.py` | Generates the SFT LaTeX results table from saved outputs. |
| `stats/dataset_tables.py` | Generates dataset summary tables. |
| `stats/dataset_figures.py` | Generates dataset figures. |
| `stats/metrics_table_utils.py` | Shared metric and table helpers. |

### Reproducing evaluation results

The evaluation scripts in `stats/` process the predictions generated by the ZS and SFT experiments and produce the corresponding result tables.

The main scripts are:

- `stats/calculate_metrics_zs.py`: computes and formats the ZS evaluation results.
- `stats/calculate_metrics_sft.py`: computes and formats the SFT evaluation results.
- `stats/dataset_tables.py`: generates dataset statistics.
- `stats/dataset_figures.py`: generates dataset visualizations.

To reproduce the reported evaluation results, the original model predictions must be available in the directory structure expected by the evaluation scripts. The scripts should be executed using the same metric configurations and preprocessing procedures described in this repository.

The statistical comparisons reported in the manuscript use paired bootstrap resampling with 1,000 resamples.

### Evaluation preprocessing

Generated text is decoded with model-specific special tokens removed, and leading and trailing whitespace is stripped. The same trimming is applied to reference answers. No project-specific lowercasing, punctuation removal, accent stripping or internal whitespace collapsing is applied.

ROUGE metrics are computed using Hugging Face Evaluate with stemming enabled (`use_stemmer=True`). BERTScore F1 is computed using the Spanish BETO model, preserving capitalization, punctuation and diacritics.

The exact BERTScore checkpoint identifier and metric configuration are defined in the evaluation scripts.

### Prompt-length handling

All zero-shot and SFT evaluation prompts are limited to 8,192 tokens by
default, configurable through `--max_seq_len`. The limit is applied after
constructing the full prompt, including instructions and model-specific chat
template tokens. Llama, Mistral, and SOLAR use left truncation (retaining the
end of the prompt); other models use their tokenizer's default truncation
side. SFT applies the same limit while preparing training and validation
examples. Generation is capped at 256 new tokens.


## Installation

The reported experiments were conducted using Python 3.12. We recommend using this version when reproducing the published results. Other Python versions may work but have not been verified as part of the reported experimental configuration.

Create an isolated environment before installing the dependencies. The training scripts require CUDA-compatible GPUs and use DeepSpeed for distributed training and memory optimization.

### Conda

```bash
git clone https://github.com/NLP-UMUTeam/Spanish-ClickbaitCorpus-2026.git
cd Spanish-ClickbaitCorpus-2026

conda create -n spanish-clickbait python=3.12
conda activate spanish-clickbait
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### venv

```bash
git clone https://github.com/NLP-UMUTeam/Spanish-ClickbaitCorpus-2026.git
cd Spanish-ClickbaitCorpus-2026

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` lists direct dependencies only. `pip` resolves their
transitive dependencies. It pins the project version of TRL to a Git commit,
so Git and access to GitHub are required during installation.

Some checkpoints used in the experiments are gated on Hugging Face. Request
access from the corresponding model page and authenticate before running an
experiment that uses one:

```bash
huggingface-cli login
```

For GPU and multi-GPU fine-tuning details, see the `code/README.md`.

### Computational environment

The experiments reported in the associated manuscript were conducted using the following infrastructure:

| Component | Configuration |
| --- | --- |
| GPUs | 3 NVIDIA L40S (48 GB each) |
| CPUs | 2 Intel Xeon Gold 6530 processors |
| RAM | 256 GB |
| Operating system | Ubuntu 22.04 |
| CUDA | 12.4 |
| Python | 3.12 |
| PyTorch | 2.10.0 |
| Transformers | 5.5.4 |
| DeepSpeed | 0.18.0 |
| Hugging Face Datasets | 4.2.0 |
| Hugging Face Evaluate | 0.4.6 |
| BERTScore | 0.3.13 |

These specifications describe the environment used for the reported experiments. Running the scripts on different hardware or software versions may require adjustments to the installation procedure, batch sizes or distributed training configuration.

The complete Python dependencies are listed in `requirements.txt`.

## Reproducing experiments

The scripts load the CSV directly and use its `_split` field. From `code/`, a
minimal zero-shot QA run is:

```bash
python zs.py \
  -p ../Spanish_ClickbaitCorpus_2026.csv \
  -m Qwen/Qwen3-4B \
  -a qa \
  --save_path ../results/zs_results
```

To fine-tune a base model on a single GPU:

```bash
deepspeed --include=localhost:0 --master_port 62000 \
  sft_base_version.py \
  --approach qa \
  --csv_path ../Spanish_ClickbaitCorpus_2026.csv \
  --save_path ../results/sft_results \
  --model_id google/gemma-2-2b \
  --train_bs 2 --eval_bs 2
```

To fine-tune an instruction-tuned model on a single GPU:

```bash
deepspeed --include=localhost:0 --master_port 62000 \
  sft_instructed_version.py \
  --approach qa \
  --csv_path ../Spanish_ClickbaitCorpus_2026.csv \
  --save_path ../results/sft_results \
  --model_id Qwen/Qwen3-4B \
  --train_bs 2 --eval_bs 2
```

The output directories are created automatically. Full model support,
arguments, prompt handling, and additional examples are documented in `code/README.md`.

## FAIR data and licence

The corpus was curated following the FAIR (Findable, Accessible,
Interoperable, Reusable) principles. A FAIR self-assessment documents the use
of persistent identifiers, standard metadata, clear access conditions, and
provenance information. The dataset and accompanying experimental code are publicly available in this repository. 
The dataset will additionally be deposited in a research-data repository to provide persistent identification and support long-term preservation.

Dataset annotations and metadata are released under the Creative Commons
Attribution-NonCommercial 4.0 International licence (CC BY-NC 4.0). Original
news articles remain subject to the copyright and terms of their respective
publishers. This resource is provided for research purposes.

## Contributions and issue reporting

Contributions that improve the documentation, reproducibility or usability of Spanish ClickbaitCorpus 2026 are welcome.

If you identify an issue with the dataset, code or experimental instructions, please open a GitHub issue describing the problem and, when applicable, the steps required to reproduce it.

For code contributions, please open a pull request explaining the proposed changes. Contributions should preserve compatibility with the existing dataset structure and experimental evaluation protocol.

Corrections affecting dataset annotations or experimental results should be clearly documented to maintain transparency and reproducibility.