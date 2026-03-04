# SFT training of Gemma 2 2b base version
deepspeed --include=localhost:0,1,2 --master_port 62000 sft_base_version.py --approach "qa" --csv_path "./Spanish_ClickbaitCorpus_2026.csv" --model_id "google/gemma-2-2b" --train_bs 8 --eval_bs 8
deepspeed --include=localhost:0,1,2 --master_port 62000 sft_base_version.py --approach "summ" --csv_path "./Spanish_ClickbaitCorpus_2026.csv" --model_id "google/gemma-2-2b" --train_bs 8 --eval_bs 8

# SFT training of Gemma 2 2b instructed version
deepspeed --include=localhost:0,1,2 --master_port 62000 sft_instructed_version.py --approach "qa" --csv_path "./Spanish_ClickbaitCorpus_2026.csv" --model_id "google/gemma-2-2b-it" --train_bs 8 --eval_bs 8
deepspeed --include=localhost:0,1,2 --master_port 62000 sft_instructed_version.py --approach "summ" --csv_path "./Spanish_ClickbaitCorpus_2026.csv" --model_id "google/gemma-2-2b-it" --train_bs 8 --eval_bs 8