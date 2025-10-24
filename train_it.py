import argparse, os, json
import pandas as pd
import torch
import evaluate
import random
import torch.distributed as dist

from datasets import load_dataset

from transformers import (
    set_seed, AutoTokenizer, AutoModelForCausalLM
)

from trl import SFTTrainer, SFTConfig
import numpy as np
from transformers import GenerationConfig

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

rouge = evaluate.load("rouge")

def build_prompt_for_generation(example, tokenizer, model_name):

    instr = example.get("instruction", "") or ""
    inp   = example.get("input", "") or ""

    user_content = instr if not inp.strip() else f"{instr}\n\n{inp}"

    if model_name in {"gemma_3_1b_it", "gemma_3_4b_it", "gemma_3_12b_it", "gemma_3_27b_it"}:
        messages = [{"role": "user", 
                     "content": [{"type": "text", "text": user_content.strip()}]
                    }]
    else:
        messages = [{"role": "user", "content": user_content.strip()}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return text

def eval_rouge_loop(model, tokenizer, model_name, val_ds, max_len=8192, gen_len=256, bs=2):
    model.eval()
    preds, refs = [], []

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.model_id in {"meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "mistralai/Mistral-7B-Instruct-v0.3", "deepseek-ai/deepseek-llm-7b-chat", "upstage/SOLAR-10.7B-v1.0"}:
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

    with torch.no_grad():

        batch_prompts, batch_refs = [], []

        for i in range(len(val_ds)):

            ex = val_ds[i]
            
            _, reference = ex.get("instruction",""), ex.get("output","")
            reference = ex.get("output","") or ""

            
            prompt_text = build_prompt_for_generation(ex, tokenizer, model_name)

            batch_prompts.append(prompt_text)
            batch_refs.append(reference.strip())

            if len(batch_prompts) == bs or i == len(val_ds)-1:

                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len
                ).to(model.device)

                input_lengths = (inputs.input_ids != tokenizer.pad_token_id).sum(dim=1)

                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=gen_len,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

                
                gens_only = []

                for seq, in_len in zip(gen_ids, input_lengths):
                    gens_only.append(seq[in_len:])

                texts = tokenizer.batch_decode(
                    torch.nn.utils.rnn.pad_sequence([g for g in gens_only], batch_first=True, padding_value=tokenizer.pad_token_id),
                    skip_special_tokens=True
                )

                preds.extend([t.strip() for t in texts])
                refs.extend([r.strip() for r in batch_refs])
                batch_prompts, batch_refs = [], []

    return preds, rouge.compute(predictions=preds, references=refs, use_stemmer=True)


def row_to_chat_text(row, tokenizer, model_name):
    
    instr = row.get("instruction", "") or ""
    inp   = row.get("input", "") or ""
    out   = row.get("output", "") or ""

    user_content = instr if not inp.strip() else f"{instr}\n\n{inp}"

    messages = [
        {"role": "user", "content": user_content.strip()},
        {"role": "assistant", "content": out.strip()},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

def add_text_field(example, tokenizer, model_name):
    return {"text": row_to_chat_text(example, tokenizer, model_name)}

def obtain_model_path(model_id):
    
    # Gemma 2
    if model_id == "google/gemma-2-2b-it":
        model_name = "gemma_2_2b_it"

    # Gemma 3
    elif model_id == "google/gemma-3-1b-it":
        model_name = "gemma_3_1b_it"
    elif model_id == "google/gemma-3-4b-it":
        model_name = "gemma_3_4b_it"

    # Llama 3.1
    elif model_id == "meta-llama/Llama-3.1-8B-Instruct":
        model_name = "llama_3_1_8b_it"

    # Llama 3.2
    elif model_id == "meta-llama/Llama-3.2-1B-Instruct":
        model_name = "llama_3_2_1b_it"
    elif model_id == "meta-llama/Llama-3.2-3B-Instruct":
        model_name = "llama_3_2_3b_it"

    # Qwen 3
    elif model_id == "Qwen/Qwen3-1.7B":
        model_name = "qwen_3_1b"
    elif model_id == "Qwen/Qwen3-4B":
        model_name = "qwen_3_4b"
    elif model_id == "Qwen/Qwen3-8B":
        model_name = "qwen_3_8b"

    # Mistral
    elif model_id == "mistralai/Mistral-7B-Instruct-v0.3":
        model_name = "mistral_v3_it"

    # Openchat
    elif model_id == "openchat/openchat-3.5-0106":
        model_name = "openchat_3_5_0106"

    # Deepkseek
    elif model_id == "deepseek-ai/deepseek-llm-7b-chat":
        model_name = "deepseek_llm_7b_it"

    # Solar
    elif model_id == "upstage/SOLAR-10.7B-Instruct-v1.0":
        model_name = "solar_10b_it"

    return model_name

def build_qa_from_csv(df):

    rows = []

    for _, row in df.iterrows():

        headline = str(row["Título"])
        context = str(row["Cuerpo"])
        answer = str(row["Respuesta"])

        rows.append({
            
            "instruction": f"Based on the news article, give a concise answer that directly addresses the question behind the headline.\n\nHeadline: {headline}",
            "input": f"Article:\n{context}",

            "output": answer.strip(),
        })

    return rows

def build_summ_from_csv(df):
    
    rows = []

    for _, row in df.iterrows():

        headline = str(row["Título"])
        context = str(row["Cuerpo"])
        answer = str(row["Respuesta"])

        rows.append({
            "instruction": f"Summarize the following news article titled '{headline}'.",

            "input": f"Article:\n{context}",

            "output": answer.strip(),
        })

    return rows

def prepare_sample_text(ex):
    instr = ex.get("instruction","")
    inp = ex.get("input","")
    out = ex.get("output","")
    return f"### Question:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
       
def main(args):

    print("Begin of SFT under " + args.approach + " approach with " + obtain_model_path(args.model_id))

    set_seed(args.seed)

    os.makedirs(args.save_path, exist_ok=True)

    df = pd.read_csv(args.csv_path)

    df_train = df[df["_split"] == "train"]
    df_val   = df[df["_split"] == "val"]
    df_test   = df[df["_split"] == "test"]

    if args.approach == "qa":
        train_rows = build_qa_from_csv(df_train)
        val_rows = build_qa_from_csv(df_val)
        test_rows = build_qa_from_csv(df_test)
    else:
        train_rows = build_summ_from_csv(df_train)
        val_rows = build_summ_from_csv(df_val)
        test_rows = build_summ_from_csv(df_test)

    if len(train_rows) == 0 or len(val_rows) == 0 or len(test_rows) == 0:
        raise ValueError("No examples generated.")

    data_dir = os.path.join(args.save_path, "data_dir")
    os.makedirs(data_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path   = os.path.join(data_dir, "val.jsonl")
    test_path   = os.path.join(data_dir, "test.jsonl")
    
    with open(train_path,"w",encoding="utf-8") as f:
        for r in train_rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    
    with open(val_path,"w",encoding="utf-8") as f:
        for r in val_rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")

    with open(test_path,"w",encoding="utf-8") as f:
        for r in test_rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
        trust_remote_code=True
    )
        
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    train_data = load_dataset("json", data_files=train_path, split="train")
    val_data   = load_dataset("json", data_files=val_path, split="train")
    test_data = load_dataset("json", data_files=test_path, split="train")
    
    train_data = train_data.map(
        add_text_field,
        fn_kwargs={"tokenizer": tokenizer, "model_name": obtain_model_path(args.model_id)}
    )

    val_data = val_data.map(
        add_text_field,
        fn_kwargs={"tokenizer": tokenizer, "model_name": obtain_model_path(args.model_id)}
    )

    test_data = test_data.map(
        add_text_field,
        fn_kwargs={"tokenizer": tokenizer, "model_name": obtain_model_path(args.model_id)}
    )

    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    num_gpus = torch.cuda.device_count()

    if local_rank != -1 and local_rank < num_gpus:
        torch.cuda.set_device(local_rank)
        
    device_map = {"": local_rank} if local_rank != -1 and local_rank < num_gpus else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map=device_map,
        trust_remote_code=True
    )

    model.config.use_cache = False

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id

    if args.model_id == "openchat/openchat-3.5-0106":
        gen_cfg = model.generation_config or GenerationConfig()
        gen_cfg.temperature = None
        gen_cfg.top_p = None
        gen_cfg.top_k = None
        gen_cfg.do_sample = False
        model.generation_config = gen_cfg

    training_args = SFTConfig(
        output_dir=os.path.join(args.save_path,"runs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.00,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-12,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        gradient_checkpointing=True,
        deepspeed="deepspeed.json",
        bf16=True,
        packing=False,

        dataset_text_field="text"
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_data, 
        eval_dataset=val_data,
        formatting_func=prepare_sample_text
    )

    try:
        trainer.train()
        
        preds, rouge_scores = eval_rouge_loop(
            model=trainer.model,
            tokenizer=tokenizer,
            model_name = obtain_model_path(args.model_id),
            val_ds=test_data
        )

        print("\n=== Results ROUGE ===")
        for k, v in rouge_scores.items():
            print(f"{k}: {v:.4f}")

        out_dir = os.path.join(args.save_path, obtain_model_path(args.model_id) + "_" + args.approach)
        os.makedirs(out_dir, exist_ok=True)

        pd.DataFrame(preds).to_csv(out_dir + "/" + obtain_model_path(args.model_id) + "_predictions.csv")

        df_rouge = pd.DataFrame([
            {"metric": k, "score": v}
            for k, v in rouge_scores.items()
        ])
        
        df_rouge.to_csv(out_dir + "/" + obtain_model_path(args.model_id) + "_metrics.csv")

        print("End of SFT under " + args.approach + " approach with " + obtain_model_path(args.model_id))

    except KeyboardInterrupt:
        pass
    finally:
        try:
            trainer.accelerator.wait_for_everyone()
        except Exception:
            pass
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", required=True, type=str, help="Approach to evaluate")
    parser.add_argument("--csv_path", required=True, type=str, help="CSV path")
    parser.add_argument("--save_path", type=str, default="./results", help="Results path")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-1b-pt")
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--train_bs", type=int, default=2)
    parser.add_argument("--eval_bs", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.000005)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="(auto) rank local")
    
    args = parser.parse_args()
    main(args)
