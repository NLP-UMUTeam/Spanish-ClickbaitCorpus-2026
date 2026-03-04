# ZS evaluation of Gemma 2 2b base version
python zs.py -p "./Spanish_ClickbaitCorpus_2026.csv" -a "qa" -m "google/gemma-2-2b"
python zs.py -p "./Spanish_ClickbaitCorpus_2026.csv" -a "summ" -m "google/gemma-2-2b"

# ZS evaluation of Gemma 2 2b instructed version
python zs.py -p "./Spanish_ClickbaitCorpus_2026.csv" -a "qa" -m "google/gemma-2-2b-it"
python zs.py -p "./Spanish_ClickbaitCorpus_2026.csv" -a "summ" -m "google/gemma-2-2b-it"