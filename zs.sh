# Gemma 2 2b base
python zs.py -p "./noticias_contenido_final.csv" -a "qa" -m "google/gemma-2-2b"
python zs.py -p "./noticias_contenido_final.csv" -a "summ" -m "google/gemma-2-2b"

# Gemma 2 2b it
python zs.py -p "../noticias_contenido_final.csv" -a "qa" -m "google/gemma-2-2b-it"
python zs.py -p "../noticias_contenido_final.csv" -a "summ" -m "google/gemma-2-2b-it"