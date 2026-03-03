# Spanish Clickbait Corpus 2026
#### Question Answering vs. Summarization for Clickbait Resolution: A Comparative Study in Spanish using the Spanish ClickbaitCorpus 2026
This paper presents a comparative study of question-answering and summarization paradigms for clickbait resolution in Spanish. We introduce the Spanish ClickbaitCorpus 2026, comprising over 1,100 news articles with clickbait-style headlines. Each headline is paired with an extractive answer span that resolves the informational gap induced by the headline. Using this corpus, we evaluate several open-source Large Language Models under both Zero-Shot and Supervised Fine-Tuning settings. We use instruction-based data configurations to assess which paradigm better captures the reasoning and factual grounding required to spoil clickbait. The results show that Question Answering consistently outperforms summarization across almost all models and settings, achieving higher ROUGE-L and ROUGE-1 scores and exhibiting greater lexical fidelity and structural coherence. These results suggest that framing clickbait resolution as a Question Answering task more effectively models the inferential process underlying reader curiosity. Future work will focus on developing a Spanish Large Language Models specifically tuned for clickbait resolution and expanding the corpus with additional multimodal sources and annotations to strengthen coverage and representativeness.

## Dataset
- noticias_contenido_final.csv

## Code for SFT experiments
- ```train_base.py```: SFT script for the base variants.
- ```train_it.py```: SFT script for the instruction-tuned variants.
- ```deepspeed.json```: Deepspeed configuration file.
- ```run.sh```: Shell script for launching SFT experiments.

## Code for ZS experiments
- ```zs.py```: Script for running Zero-Shot evaluations.
- ```zs.sh```: Shell script for launching ZS experiments.
