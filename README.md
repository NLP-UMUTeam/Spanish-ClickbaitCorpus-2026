# Spanish Clickbait Corpus 2026

## Question Answering vs. Summarization for Clickbait Resolution: A Comparative Study in Spanish using the Spanish ClickbaitCorpus 2026
**DOI will be added upon paper acceptance.**

Spanish ClickbaitCorpus 2026 is a Spanish-language dataset for clickbait
resolution constructed from online news articles. Each instance pairs a
clickbait-style headline with an extractive answer span from the article
body that resolves the informational gap introduced by the headline.

The corpus was introduced in the associated scientific publication and is
released following the FAIR data principles to facilitate reproducibility,
transparency and reuse in future research on clickbait detection,
resolution and related natural language understanding tasks.

### Highlights
- Spanish ClickbaitCorpus 2026: new dataset for clickbait resolution in Spanish.

- Systematic comparison of QA and summarization paradigms using modern LLMs.

- Experiments conducted in Zero-Shot and Supervised Fine-Tuning settings.

- Evaluation includes ROUGE, BERTScore and a hybrid metric proposed in this work.

- QA formulation consistently outperforms summarization.

### Authors

- **Tomás Bernal-Beltrán** — University of Murcia  
  [Google Scholar](https://scholar.google.com/citations?user=0bTUxQEAAAAJ) · [ORCID](https://orcid.org/0009-0006-6971-1435)

- **Ronghao Pan** — University of Murcia  
  [Google Scholar](https://scholar.google.com/citations?user=80lntLMAAAAJ) · [ORCID](https://orcid.org/0009-0008-7317-7145)

- **José Antonio García-Díaz** — University of Murcia  
  [Google Scholar](https://scholar.google.com/citations?user=ek7NIYUAAAAJ) · [ORCID](https://orcid.org/0000-0002-3651-2660)

- **Rafael Valencia-García** — University of Murcia  
  [Google Scholar](https://scholar.google.com/citations?user=GLpBPNMAAAAJ) · [ORCID](https://orcid.org/0000-0003-2457-1791)  

> **Affiliations:**  
> \* *Departamento de Informática y Sistemas, Universidad de Murcia, Campus de Espinardo, 30100, Murcia, Spain*

### Publication
**Publication information will be added upon paper acceptance.**

### Acknowledgments
**Acknowledgments will be added upon paper acceptance.**

### Citation
**Citation information will be added upon paper acceptance.**

### Abstract
This paper presents a systematic comparison of Question Answering (QA) and summarization paradigms for clickbait resolution in Spanish. We introduce Spanish ClickbaitCorpus 2026, a new dataset comprising over 1,100 news articles with clickbait-style headlines, each paired with an extractive answer span designed to resolve the informational gap induced by the headline. Using this corpus, we evaluate several open-source Large Language Models under both Zero-Shot (ZS) and Supervised Fine-Tuning (SFT) settings. Performance is assessed using complementary lexical and semantic metrics, including ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum, BERTScore F1 and a Hybrid indicator that combines structural and contextual alignment. Statistical significance is evaluated through paired bootstrap resampling. Across models and experimental conditions, QA consistently outperforms summarization. The advantage is particularly pronounced in the ZS setting and remains statistically reliable after SFT, although the performance gap narrows with task-specific adaptation. These findings suggest that framing clickbait resolution as an explicit QA problem provides a stronger structural inductive bias, promoting span-level grounding and more stable resolution of the informational gap. Future work will focus on developing a Spanish Large Language Models specifically tuned for clickbait resolution and expanding the corpus with additional multimodal sources and annotations to strengthen coverage and representativeness.

## Dataset
The dataset is distributed in the following file: Spanish_ClickbaitCorpus_2026.csv

This file contains the Spanish ClickbaitCorpus 2026 dataset used in the
experiments described in the associated paper. Each row corresponds to a news article containing a clickbait-style headline and an extractive answer span that resolves the informational gap introduced by the headline.

The corpus was constructed from Spanish online news articles and is intended
for research on clickbait resolution, question answering, and automatic
content understanding.

**The dataset will be publicly released upon acceptance of the associated paper.**

### Dataset statistics
The table below summarizes the structure of the Spanish ClickbaitCorpus 2026 dataset. 
The corpus contains a total of 1,148 news articles with clickbait-style headlines 
paired with extractive answer spans that resolve the informational gap introduced 
by the headline. The dataset is divided into training, validation and test splits 
to support both supervised fine-tuning and evaluation experiments.

In addition to the number of instances per split, the table also reports the average 
length of headlines, article bodies and answer spans, measured in number of words. 
These statistics provide an overview of the typical size of the inputs and target 
responses used in the clickbait resolution task.

| | Train | Validation | Test |
|----------|------:|-----:|------:|
| Number of Articles |  918  | 115  |  115  |
| Avg. words in headlines |  16  | 16  | 17  |
| Avg. words in articles |   444  |  446   |   418  |
| Avg. words in Answers |  12  | 12  |  12  |

The following table shows the distribution of article categories across the
training, validation and test splits of the Spanish ClickbaitCorpus 2026.
The categories correspond to the main topic of each news article.

| Class | Train | Validation | Test | Total |
|----------|------:|-----:|------:|------:|
| Actualidad | 272 | 44 | 38 | 354 |
| Deportes | 119 | 8 | 11 | 138 |
| Celebrities | 79 | 10 | 14 | 103 |
| Salud | 78 | 9 | 7 | 94 |
| Gastronomía | 59 | 7 | 6 | 72 |
| Cine | 61 | 3 | 6 | 70 |
| Turismo | 43 | 7 | 6 | 56 |
| Economía | 41 | 4 | 6 | 51 |
| Legislación | 28 | 2 | 3 | 33 |
| Tecnología | 25 | 4 | 4 | 33 |
| Clima | 27 | 2 | 3 | 32 |
| Televisión | 16 | 4 | 2 | 22 |
| Seguridad | 15 | 4 | 1 | 20 |
| Ciencia | 12 | 2 | 4 | 18 |
| Videojuegos | 15 | 1 | 1 | 17 |
| Lenguaje | 8 | 3 | 1 | 12 |
| Música | 10 | 0 | 2 | 12 |
| Automoción | 8 | 1 | 0 | 9 |
| Belleza | 2 | 0 | 0 | 2 |

### Data fields

The public dataset includes the following fields:

- `URL`: URL of the original news article from which the example was collected.
- `Periódico`: Name of the newspaper or media outlet that published the article.
- `Título`: Clickbait-style headline of the news article.
- `Redactor`: Author or journalist associated with the article (when available).
- `Cuerpo`: Full text of the news article body.
- `Respuesta`: Extractive answer span that resolves the informational gap introduced by the headline.
- `Clase`: Topical category assigned to the article (e.g., actualidad, deportes, salud, etc.).
- `Multi Clase`: Multi-label category information when multiple topics are associated with the article.
- `_split`: Dataset split used for experiments (`train`, `validation`, `test`).

### FAIR data and self-assessment
This dataset has been curated following the FAIR (Findable, Accessible, Interoperable, Reusable) data principles.

A FAIR self-assessment has been conducted using the FAIR Data Self-Assessment Tool (FAISS), documenting the use of persistent identifiers, standard metadata, clear access conditions and detailed provenance information.

The dataset is currently available through this GitHub repository and will be archived in a public research data repository upon acceptance of the associated publication to ensure long-term accessibility and reproducibility.

The resource is intended to support future research on clickbait resolution, question answering, and automatic understanding of sensationalist headlines in Spanish news media.

## Experimental pipeline

The workflow of this study can be summarized as follows. 
First, we construct and annotate the **Spanish ClickbaitCorpus 2026**, a dataset composed of Spanish news articles with clickbait-style headlines paired with extractive answer spans that resolve the informational gap introduced by the headline.

Next, the collected articles are processed and organized into training, validation and test splits. The training split is used for supervised fine-tuning experiments, while the validation and test sets are used to monitor model performance and conduct the final evaluation.

In the experimental stage, we evaluate two alternative formulations of the clickbait resolution task: **Question Answering (QA)** and **headline-conditioned summarization**. Both paradigms are tested using several open-source decoder-only Large Language Models under two experimental settings: **Zero-Shot (ZS)** inference and **Supervised Fine-Tuning (SFT)**.

Finally, model outputs are evaluated using complementary lexical and semantic metrics, including **ROUGE** and **BERTScore**, as well as a **hybrid metric proposed in this work** that combines structural and contextual alignment. This experimental framework enables a systematic comparison between QA and summarization paradigms for resolving clickbait headlines.

### Evaluation

The table below summarizes the performance of the evaluated Large Language Models under both **ZS** and **SFT** settings for the clickbait resolution task. We compare two alternative task formulations: **QA** and **headline-conditioned summarization**.

| Setting | Best QA Model | Best Summarization Model | Main Observation |
|--------|---------------|--------------------------|------------------|
| ZS | Gemma 3 4B (IT version) | Gemma 3 4B (IT version) | QA clearly outperforms summarization |
| SFT | Gemma 2 2B (IT version) | OpenChat 3.5 (IT version) | QA still maintains an advantage |

Across the experiments, QA consistently achieves stronger performance than summarization, particularly in the ZS setting where the difference between both paradigms is more pronounced. This suggests that framing the headline as an implicit question provides a stronger inductive bias for resolving the informational gap introduced by clickbait headlines.

After task-specific fine-tuning, the performance gap becomes smaller, as summarization models learn to better approximate the target answer spans. However, QA-based models still maintain a measurable advantage across the main evaluation metrics, including **ROUGE-L** and the
**Hybrid metric proposed in this work**, which combines lexical alignment and semantic similarity.

To assess whether these differences are statistically meaningful, we conducted **paired bootstrap resampling tests with 1,000 resamples** on the best-performing configurations in each setting. The analysis confirms that the performance advantage of QA over summarization is
statistically significant for both **ROUGE-L** and the **Hybrid metric** in the ZS and SFT conditions.

These results suggest that clickbait resolution is more naturally modeled as a **targeted information retrieval problem**, where the goal is to recover a minimal answer span that resolves the curiosity gap introduced by the headline, rather than as a generic summarization task.

## Reproducibility

This repository includes the code used to reproduce the experiments reported in the associated paper. The provided scripts allow running both **ZS** evaluations and **SFT**
experiments with the models evaluated in the study.

The code covers the main stages of the experimental pipeline, including model loading, prompt construction, SFT, inference and generation of model predictions for evaluation. Shell scripts are also provided to facilitate launching the experiments in a reproducible way.

### Code for SFT experiments

- `sft_base_version.py`: Script for running SFT with the **base variants** of the evaluated LLMs.
- `sft_instructed_version.py`: Script for running SFT with the **instruction-tuned variants**.
- `deepspeed.json`: DeepSpeed configuration file used for efficient training.
- `sft_experiments.sh`: Shell script for launching the SFT experiments.

### Code for ZS experiments

- `zs.py`: Script for running **ZS inference** using the evaluated LLMs.
- `zs_experiments.sh`: Shell script for launching the Zero-Shot experiments.

## License

The dataset annotations and metadata are distributed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.

The original news articles remain subject to the copyright and terms of use of their respective publishers. This repository only distributes the annotations and metadata necessary to reproduce the experiments described in the associated publication.

The dataset is released for research purposes only.