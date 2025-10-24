# Gemma 2 2b base
deepspeed --include=localhost:0,1,2 --master_port 62000 train_base.py --approach "qa" --csv_path "./noticias_contenido_final.csv" --model_id "google/gemma-2-2b" --train_bs 8 --eval_bs 8
deepspeed --include=localhost:0,1,2 --master_port 62000 train_base.py --approach "summ" --csv_path "./noticias_contenido_final.csv" --model_id "google/gemma-2-2b" --train_bs 8 --eval_bs 8

# Gemma 2 2b it
deepspeed --include=localhost:0,1,2 --master_port 62000 train_it.py --approach "qa" --csv_path "./noticias_contenido_final.csv" --model_id "google/gemma-2-2b-it" --train_bs 8 --eval_bs 8
deepspeed --include=localhost:0,1,2 --master_port 62000 train_it.py --approach "summ" --csv_path "./noticias_contenido_final.csv" --model_id "google/gemma-2-2b-it" --train_bs 8 --eval_bs 8