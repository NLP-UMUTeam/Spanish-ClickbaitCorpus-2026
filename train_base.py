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

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

rouge = evaluate.load("rouge")


def to_text(example):

    return example['instruction'], example['input']

def eval_rouge_loop(model, tokenizer, val_ds, max_len=8192, gen_len=256, bs=2):

    model.eval()
    preds, refs = [], []

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.model_id in {"meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B", "mistralai/Mistral-7B-v0.3", "deepseek-ai/deepseek-llm-7b-base", "upstage/SOLAR-10.7B-v1.0"}:
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        
    with torch.no_grad():
        batch = []
        for i in range(len(val_ds)):
            prompt, reference = to_text(val_ds[i])
            batch.append((prompt, reference))
            if len(batch) == bs or i == len(val_ds)-1:
                inputs = tokenizer([p for p, _ in batch],
                                   return_tensors="pt",
                                   padding=True, truncation=True,
                                   max_length=max_len).to(model.device)

                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=gen_len,
                    do_sample=True
                )
                texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

                preds.extend([t.strip() for t in texts])
                refs.extend([r.strip() for _, r in batch])
                batch = []

    return preds, rouge.compute(predictions=preds, references=refs, use_stemmer=True)

def obtain_model_path(model_id):
    
    # Gemma 2
    if model_id == "google/gemma-2-2b":
        model_name = "gemma_2_2b"

    # Gemma 3
    elif model_id == "google/gemma-3-1b-pt":
        model_name = "gemma_3_1b"
    elif model_id == "google/gemma-3-4b-pt":
        model_name = "gemma_3_4b"

    # Llama 3.1
    elif model_id == "meta-llama/Llama-3.1-8B":
        model_name = "llama_3_1_8b"

    # Llama 3.2
    elif model_id == "meta-llama/Llama-3.2-1B":
        model_name = "llama_3_2_1b"
    elif model_id == "meta-llama/Llama-3.2-3B":
        model_name = "llama_3_2_3b"

    # Mistral
    elif model_id == "mistralai/Mistral-7B-v0.3":
        model_name = "mistral_v3"

    # Deepkseek
    elif model_id == "deepseek-ai/deepseek-llm-7b-base":
        model_name = "deepseek_llm_7b"

    # Solar
    elif model_id == "upstage/SOLAR-10.7B-v1.0":
        model_name = "solar_10b"

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
        use_fast=True
    )

    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    train_data = load_dataset("json", data_files=train_path, split="train")
    val_data   = load_dataset("json", data_files=val_path, split="train")
    test_data = load_dataset("json", data_files=test_path, split="train")
    
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
