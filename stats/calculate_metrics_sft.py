import argparse
from pathlib import Path

from metrics_table_utils import DATASET_PATH, REPOSITORY_ROOT, TABLES_DIR, write_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the supervised fine-tuning LaTeX results table.")
    parser.add_argument("--skip-bertscore", action="store_true", help="Generate a structural draft without BERTScore or Hybrid values.")
    parser.add_argument("--results-dir", type=Path, default=REPOSITORY_ROOT / "results" / "sft_results", help="Directory containing SFT result CSV files.")
    parser.add_argument("--csv-path", type=Path, default=DATASET_PATH, help="Dataset CSV used as the test-set reference.")
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR, help="Directory for the generated LaTeX table.")
    args = parser.parse_args()
    print(write_table("sft", args.skip_bertscore, args.results_dir, args.csv_path, args.output_dir))
