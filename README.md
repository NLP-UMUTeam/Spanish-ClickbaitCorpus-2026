# Spanish ClickbaitCorpus 2026

## Question Answering vs. Summarization for Clickbait Resolution in Spanish

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

Across the reported experiments, QA outperforms headline-conditioned
summarization, especially in the zero-shot setting. The gap narrows after
fine-tuning, but QA retains an advantage on the main metrics.

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

## Installation

Use Python 3.10 or later and create an isolated environment. The training
scripts use CUDA and DeepSpeed; install a PyTorch build compatible with your
GPU driver and CUDA runtime before, or together with, the project
dependencies.

### Conda

```bash
git clone https://github.com/NLP-UMUTeam/Spanish-ClickbaitCorpus-2026.git
cd Spanish-ClickbaitCorpus-2026

conda create -n spanish-clickbait python=3.10
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

For a single-GPU instruction-tuned SFT run:

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

### Evaluation preprocessing

Generated text is decoded with special tokens removed and leading/trailing
whitespace is stripped. The same trimming is applied to reference answers;
no project-specific lowercasing, punctuation removal, accent stripping, or
internal-whitespace collapsing is applied. ROUGE is computed with Hugging
Face Evaluate's ROUGE implementation and stemming enabled. BERTScore is
computed on the otherwise unmodified generated and reference texts, excluding
examples whose prediction or reference is empty after trimming.

### Prompt-length handling

All zero-shot and SFT evaluation prompts are limited to 8,192 tokens by
default, configurable through `--max_seq_len`. The limit is applied after
constructing the full prompt, including instructions and model-specific chat
template tokens. Llama, Mistral, and SOLAR use left truncation (retaining the
end of the prompt); other models use their tokenizer's default truncation
side. SFT applies the same limit while preparing training and validation
examples. Generation is capped at 256 new tokens.

## FAIR data and licence

The corpus was curated following the FAIR (Findable, Accessible,
Interoperable, Reusable) principles. A FAIR self-assessment documents the use
of persistent identifiers, standard metadata, clear access conditions, and
provenance information. The dataset will be archived in a public research-data
repository after acceptance of the associated publication.

Dataset annotations and metadata are released under the Creative Commons
Attribution-NonCommercial 4.0 International licence (CC BY-NC 4.0). Original
news articles remain subject to the copyright and terms of their respective
publishers. This resource is provided for research purposes.
