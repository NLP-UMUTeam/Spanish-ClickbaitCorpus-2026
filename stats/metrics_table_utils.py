import csv
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPOSITORY_ROOT / "Spanish_ClickbaitCorpus_2026.csv"
TABLES_DIR = Path(__file__).resolve().parent / "results_tables"
MODELS = ["gemma_2_2b", "gemma_3_1b", "gemma_3_4b", "llama_3_1_8b", "llama_3_2_1b", "llama_3_2_3b", "qwen_3_1b", "qwen_3_4b", "qwen_3_8b", "mistral_v3", "openchat_3_5_0106", "deepseek_llm_7b", "solar_10_v1"]
MODEL_LABELS = {
    "gemma_2_2b": "Gemma 2 2b", "gemma_3_1b": "Gemma 3 1b", "gemma_3_4b": "Gemma 3 4b",
    "llama_3_1_8b": "LLaMA 3.1 8b", "llama_3_2_1b": "LLaMA 3.2 1b", "llama_3_2_3b": "LLaMA 3.2 3b",
    "qwen_3_1b": "Qwen 3 1.7b", "qwen_3_4b": "Qwen 3 4b", "qwen_3_8b": "Qwen 3 8b",
    "mistral_v3": "Mistral 7b v0.3", "openchat_3_5_0106": "OpenChat 3.5 0106",
    "deepseek_llm_7b": "DeepSeek llm 7b", "solar_10_v1": "SOLAR 10 v1",
}
MODEL_ALIASES = {"solar_10b": "solar_10_v1"}
METRICS = ("rouge1", "rouge2", "rougeL", "rougeLsum", "BertScore_f1", "hibrida")
FILE_PATTERN = re.compile(r"^(?P<prefix>.+)_(?P<task>qa|summ)_(?P<kind>metrics|responses|predictions)\.csv$")
HIGHLIGHTS = {
    "zs": {
        "qa": ("gemma_3_4b", {"rougeL", "hibrida"}, "bold"),
        "summ": ("gemma_3_4b", {"rougeL", "hibrida"}, "underline"),
    },
    "sft": {
        "qa": ("gemma_2_2b", {"rougeL", "hibrida"}, "bold"),
        "summ": ("openchat_3_5_0106", {"rougeLsum", "hibrida"}, "underline"),
    },
}


def read_rouge(path):
    values = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                values[row["metric"]] = float(row["score"])
            except (KeyError, TypeError, ValueError):
                continue
    return values


def load_references(csv_path=DATASET_PATH):
    data = pd.read_csv(csv_path)
    return data.loc[data["_split"] == "test", ["Title", "Response"]].reset_index(drop=True)


def load_predictions(path, setting):
    frame = pd.read_csv(path)
    if setting == "zs":
        if not {"Title", "Response"}.issubset(frame.columns):
            raise ValueError(f"Unexpected ZS response schema in {path}")
        return frame
    columns = [column for column in frame.columns if not str(column).startswith("Unnamed")]
    if not columns:
        raise ValueError(f"No prediction column found in {path}")
    return frame[columns[-1]].rename("Response").reset_index(drop=True)


def calculate_bertscore(path, setting, scorer, references):
    predictions = load_predictions(path, setting)
    if setting == "zs":
        merged = references.merge(predictions, on="Title", how="left", suffixes=("_gold", "_model"))
        gold, generated = merged["Response_gold"], merged["Response_model"]
    else:
        if len(predictions) != len(references):
            raise ValueError(f"Prediction count ({len(predictions)}) does not match test-set size ({len(references)}) in {path}")
        gold, generated = references["Response"], predictions
    valid = pd.DataFrame({"gold": gold, "generated": generated}).fillna("")
    valid = valid[(valid["gold"].astype(str).str.strip() != "") & (valid["generated"].astype(str).str.strip() != "")]
    if valid.empty:
        return None
    _, _, f1 = scorer.score(valid["generated"].astype(str).tolist(), valid["gold"].astype(str).tolist())
    return float(f1.mean())


def split_prefix(prefix):
    if prefix.endswith("_it"):
        model, variant = prefix[:-3], "instructed"
    else:
        model, variant = prefix, "base"
    return MODEL_ALIASES.get(model, model), variant


def collect_results(results_dir, setting, csv_path=DATASET_PATH, skip_bertscore=False):
    artifacts = defaultdict(dict)
    for path in results_dir.glob("*.csv"):
        match = FILE_PATTERN.match(path.name)
        if not match:
            continue
        model, variant = split_prefix(match["prefix"])
        if model in MODELS:
            artifacts[(model, variant, match["task"])][match["kind"]] = path

    scorer = None
    references = load_references(csv_path)
    if artifacts and not skip_bertscore:
        from bert_score import BERTScorer
        scorer = BERTScorer(model_type="dccuchile/bert-base-spanish-wwm-cased", num_layers=12, lang="es", rescale_with_baseline=False)

    results = defaultdict(lambda: defaultdict(dict))
    response_kind = "responses" if setting == "zs" else "predictions"
    for (model, variant, task), files in artifacts.items():
        values = read_rouge(files["metrics"]) if "metrics" in files else {}
        if scorer is not None and response_kind in files:
            values["BertScore_f1"] = calculate_bertscore(files[response_kind], setting, scorer, references)
        if values.get("BertScore_f1") is not None and values.get("rougeL") is not None:
            values["hibrida"] = (values["BertScore_f1"] + values["rougeL"]) / 2
        results[model][variant][task] = values
    return results


def format_value(value, style=None):
    if value is None:
        return "-- --"
    text = f"{value * 100:05.2f}"
    if style == "bold":
        return rf"\textbf{{{text}}}"
    if style == "underline":
        return rf"\underline{{{text}}}"
    return text


def render_row(results, model, variant, highlights):
    qa = results.get(model, {}).get(variant, {}).get("qa", {})
    summ = results.get(model, {}).get(variant, {}).get("summ", {})
    qa_model, qa_metrics, qa_style = highlights["qa"]
    summ_model, summ_metrics, summ_style = highlights["summ"]
    qa_winner = variant == "instructed" and model == qa_model
    summ_winner = variant == "instructed" and model == summ_model
    label = MODEL_LABELS[model]
    if qa_winner and summ_winner:
        label = rf"\underline{{\textbf{{{label}}}}}"
    elif qa_winner:
        label = rf"\textbf{{{label}}}"
    elif summ_winner:
        label = rf"\underline{{{label}}}"
    qa_values = [format_value(qa.get(metric), qa_style if qa_winner and metric in qa_metrics else None) for metric in METRICS]
    summ_values = [format_value(summ.get(metric), summ_style if summ_winner and metric in summ_metrics else None) for metric in METRICS]
    return " & ".join([label, *qa_values, *summ_values]) + r" \\"


def render_table(results, setting):
    highlights = HIGHLIGHTS[setting]
    setting_label = "ZS" if setting == "zs" else "SFT"
    label = "tab:results-zs" if setting == "zs" else "tab:results-sft"
    columns = r"\textbf{Model} & \textbf{R1} & \textbf{R2} & \textbf{RL} & \textbf{RL\_Sum} & \textbf{BS} & \textbf{Hybrid} & \textbf{R1} & \textbf{R2} & \textbf{RL} & \textbf{RL\_Sum} & \textbf{BS} & \textbf{Hybrid} \\"
    section = lambda name: [r"\midrule", rf"\multicolumn{{13}}{{c}}{{\textbf{{{name}}}}}  \\", "", r"\midrule", r" & \multicolumn{6}{c|}{\textbf{QA}} & \multicolumn{6}{c}{\textbf{SUMMARIZATION}} \\", r"\midrule", "", columns, r"\midrule", ""]
    caption = (f"{setting_label} performance for QA and summarization across base and instruction-tuned variants. Scores are reported for ROUGE-1 (R1), ROUGE-2 (R2), ROUGE-L (RL), ROUGE-Lsum (RL\\_Sum), BERTScore F1 (BS) and the Hybrid metric (Hybrid). Values are expressed as percentages. For each formulation, the best QA result is highlighted in bold, while the best summarization result is underlined. Dashes (--) indicate configurations that were not evaluated. Specifically, only instruction-tuned checkpoints were included for Qwen3, OpenChat 3.5 and DeepSeek LLM; their base-model configurations were excluded from the experimental selection.")
    lines = [r"\begin{table}[ht]", r"\centering", r"\scriptsize", r"\begin{tabular}{lcccccc|cccccc}", "", r"\toprule", r"\multicolumn{13}{c}{\textbf{Base version of the LLMs}}  \\", "", *section("" )[3:], *[render_row(results, model, "base", highlights) for model in MODELS], "", *section("Instructed version of the LLMs"), *[render_row(results, model, "instructed", highlights) for model in MODELS], r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}", ""]
    return "\n".join(lines)


def write_table(
    setting,
    skip_bertscore=False,
    results_dir=None,
    csv_path=DATASET_PATH,
    output_dir=TABLES_DIR,
):
    if results_dir is None:
        results_dir = REPOSITORY_ROOT / "results" / f"{setting}_results"
    results = collect_results(results_dir, setting, csv_path, skip_bertscore)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"results_{setting}.tex"
    output_path.write_text(render_table(results, setting), encoding="utf-8")
    return output_path
