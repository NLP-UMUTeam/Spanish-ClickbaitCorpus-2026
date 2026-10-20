# Prompt templates

This directory contains the prompt templates used by the zero-shot and
supervised fine-tuning scripts for clickbait-resolution experiments.

| File | Task | Expected output |
| --- | --- | --- |
| `zero_shot_qa_prompt.txt` | Question answering (QA) | The shortest extractive Spanish span from the article that resolves the headline. |
| `zero_shot_summarization_prompt.txt` | Headline-conditioned summarization | One concise Spanish sentence explaining the factual information behind the headline. |
| `sft_qa_prompt.txt` | Supervised QA | A concise answer to the question implicit in the headline. |
| `sft_summarization_prompt.txt` | Supervised headline-conditioned summarization | A concise summary that resolves the headline. |

## Template variables

All templates contain the following headline placeholder, which the relevant
script replaces for each dataset example:

| Placeholder | Replaced with |
| --- | --- |
| `{article_headline}` | The article headline from the `Title` field. |

The zero-shot templates also contain `{article_body}`, which `zs.py` replaces
with the article body. The SFT scripts add the `Body` field as the training
input after rendering their prompt template.

Keep these placeholder names unchanged. Edit the surrounding instructions only
when intentionally changing the experimental prompt.
