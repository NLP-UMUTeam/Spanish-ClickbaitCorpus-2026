# Dataset and result statistics

This directory contains the scripts that generate the dataset tables and
figures, and the LaTeX performance tables for the zero-shot (ZS) and
supervised fine-tuning (SFT) experiments. Run all commands below from this
directory.

## Files

| Script | Input | Output |
| --- | --- | --- |
| `dataset_tables.py` | `../Spanish_ClickbaitCorpus_2026.csv` | Three LaTeX tables in `dataset_tables/`. |
| `dataset_figures.py` | `../Spanish_ClickbaitCorpus_2026.csv` | Three PDF figures in `dataset_figures/`. |
| `calculate_metrics_zs.py` | CSV files in `../results/zs_results/` | `results_tables/results_zs.tex`. |
| `calculate_metrics_sft.py` | CSV files in `../results/sft_results/` | `results_tables/results_sft.tex`. |
| `metrics_table_utils.py` | Shared utility module for the result-table scripts. | — |

## Dataset tables

```bash
python dataset_tables.py
```

Use `--output-dir` to select where the tables are saved and `--csv-path` to
select a different input dataset:

```bash
python dataset_tables.py \
  --csv-path /path/to/dataset.csv \
  --output-dir /path/to/dataset_tables
```

The script validates the required columns and generates the following files in
`dataset_tables/`:

| File | Contents |
| --- | --- |
| `dataset_statistics.tex` | Split sizes, text-length statistics, unique sources, and largest-source share. |
| `geographical_distribution.tex` | Counts of sources and articles from Spain and Latin America. |
| `category_region_distribution.tex` | Article counts by normalized category and geographical region. |

The scripts translate the Spanish labels in the source CSV into English for
the generated tables and figures. Equivalent category variants are normalized
in the source CSV itself.

## Dataset figures

```bash
python dataset_figures.py
```

Use `--output-dir` to select where the figures are saved and `--csv-path` to
select a different input dataset:

```bash
python dataset_figures.py \
  --csv-path /path/to/dataset.csv \
  --output-dir /path/to/dataset_figures
```

The script reads the same corpus and writes these PDF figures to
`dataset_figures/`:

| File | Contents |
| --- | --- |
| `Fig1.pdf` | Article distribution across categories, grouped by region. |
| `Fig2.pdf` | Distribution of unique news outlets by region. |
| `Fig3.pdf` | Article counts for the 19 most frequent news outlets; the remainder are aggregated. |

## Result tables

The result-table scripts scan the CSV artifacts in `../results/zs_results/`
and `../results/sft_results/`. They combine the saved ROUGE metrics with
BERTScore F1 and the hybrid score, then write paper-ready LaTeX tables to
`results_tables/`.

```bash
# Zero-shot results
python calculate_metrics_zs.py

# Supervised fine-tuning results
python calculate_metrics_sft.py
```

Use `--results-dir` to select the input result files, `--csv-path` to select
the dataset containing the test references, and `--output-dir` to select where
the LaTeX table is written:

```bash
python calculate_metrics_zs.py \
  --results-dir /path/to/zs_results \
  --csv-path /path/to/dataset.csv \
  --output-dir /path/to/results_tables

python calculate_metrics_sft.py \
  --results-dir /path/to/sft_results \
  --csv-path /path/to/dataset.csv \
  --output-dir /path/to/results_tables
```

### Expected result files

The scripts recognise filenames in the following forms, where `<model>` is the
normalized model name and `<approach>` is `qa` or `summ`:

```text
../results/zs_results/<model>_<approach>_metrics.csv
../results/zs_results/<model>_<approach>_responses.csv
../results/sft_results/<model>_<approach>_metrics.csv
../results/sft_results/<model>_<approach>_predictions.csv
```

ZS response files must contain `Title` and `Response` columns. SFT prediction
files contain a `Response` column and are matched to the test split by row
order. The reference answers always come from the `test` partition of
`../Spanish_ClickbaitCorpus_2026.csv`.

### Metrics

The generated tables report:

- ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum from each run's metrics file.
- BERTScore F1, recalculated with
  `dccuchile/bert-base-spanish-wwm-cased`.
- Hybrid score: the arithmetic mean of BERTScore F1 and ROUGE-L.

The experiment scripts strip leading and trailing whitespace from predictions
and references before ROUGE, but do not apply project-specific lowercasing,
punctuation removal, accent stripping, or internal-whitespace normalization.
ROUGE uses stemming; BERTScore is recalculated from the otherwise unmodified
saved texts after excluding empty prediction/reference pairs.

BERTScore downloads its model on first use. To generate a structural table
without downloading or calculating BERTScore and the hybrid score, pass
`--skip-bertscore`:

```bash
python calculate_metrics_zs.py --skip-bertscore
python calculate_metrics_sft.py --skip-bertscore
```

The tables use bold and underline formatting to mark the configurations
highlighted in the reported experiments. A `-- --` entry indicates that the
corresponding model, variant, or task was not evaluated.

## Reproducibility notes

- Each script accepts `--csv-path` and `--output-dir`; the result-table
  scripts also accept `--results-dir`. The commands shown above use the
  repository defaults when these options are omitted.
- Dataset scripts overwrite the generated `.tex` or `.pdf` files in the
  selected output directory.
- Result-table scripts overwrite `results_zs.tex` or `results_sft.tex` in the
  selected output directory.
- Run the ZS and SFT experiment scripts first if their expected result CSVs
  are not present. See [`../code/README.md`](../code/README.md).
