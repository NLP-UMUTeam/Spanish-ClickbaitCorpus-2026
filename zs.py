from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

import torch
import random
import argparse
import csv
import gc
import os
import evaluate

import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

SEED = 42
MAX_NEW_TOKENS = 256
SISTEM_PROMPT = "You are now an Artificial Intelligence specialized in debunking sensationalist or clickbait headlines. Please follow the user's instructions as precisely as possible."

# Fijar semilla para reproducibilidad
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

parser = argparse.ArgumentParser()

parser.add_argument('-p', type=str, help='CSV path', required=True)
parser.add_argument('-m', type=str, help='Model to evaluate', required=True)
parser.add_argument('-a', type=str, help='Approach to evaluate', required=True)

args = parser.parse_args()

rouge = evaluate.load("rouge")

# Gemma 2 base
if args.m == "google/gemma-2-2b":
    model_name = "gemma_2_2b"

# Gemma 2 it
elif args.m == "google/gemma-2-2b-it":
    model_name = "gemma_2_2b_it"

# Gemma 3 base
elif args.m == "google/gemma-3-1b-pt":
    model_name = "gemma_3_1b"
elif args.m == "google/gemma-3-4b-pt":
    model_name = "gemma_3_4b"
    
# Gemma 3 it
elif args.m == "google/gemma-3-1b-it":
    model_name = "gemma_3_1b_it"
elif args.m == "google/gemma-3-4b-it":
    model_name = "gemma_3_4b_it"

# Llama 3.x base
elif args.m == "meta-llama/Llama-3.1-8B":
    model_name = "llama_3_1_8b"
elif args.m == "meta-llama/Llama-3.2-1B":
    model_name = "llama_3_2_1b"
elif args.m == "meta-llama/Llama-3.2-3B":
    model_name = "llama_3_2_3b"

# Llama 3.x it
elif args.m == "meta-llama/Llama-3.1-8B-Instruct":
    model_name = "llama_3_1_8b_it"
elif args.m == "meta-llama/Llama-3.2-1B-Instruct":
    model_name = "llama_3_2_1b_it"
elif args.m == "meta-llama/Llama-3.2-3B-Instruct":
    model_name = "llama_3_2_3b_it"

# Qwen 3
elif args.m == "Qwen/Qwen3-1.7B":
    model_name = "qwen_3_1b"
elif args.m == "Qwen/Qwen3-4B":
    model_name = "qwen_3_4b"
elif args.m == "Qwen/Qwen3-8B":
    model_name = "qwen_3_8b"

# Mistral
elif args.m == "mistralai/Mistral-7B-v0.3":
    model_name = "mistral_v3"
elif args.m == "mistralai/Mistral-7B-Instruct-v0.3":
    model_name = "mistral_v3_it"
elif args.m == "openchat/openchat-3.5-0106":
    model_name = "openchat_3_5_0106"

# Deepseek
elif args.m == "deepseek-ai/deepseek-llm-7b-chat":
    model_name = "deepseek_llm_7b_it"

# Solar
elif args.m == "upstage/SOLAR-10.7B-v1.0":
    model_name = "solar_10_v1"
elif args.m == "upstage/SOLAR-10.7B-Instruct-v1.0":
    model_name = "solar_10_v1_it"
    
data = pd.read_csv(args.p)

data = data[data["_split"] == "test"]

def generate_response(model_path, prompt, model, tokenizer):

    # Gemma 2 y 3 base
    if model_path in {"google/gemma-2-2b", "google/gemma-3-1b-pt", "google/gemma-3-4b-pt"}:
        
        inputs = tokenizer(SISTEM_PROMPT + "\n" + prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=False)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens = True).strip().split("\n")[0]

    # Gemma 2 it
    elif model_path in {"google/gemma-2-2b-it"}:

        conversation = [
            {"role": "user", "content": SISTEM_PROMPT + "\n" + prompt},
        ]

        inputs = tokenizer.apply_chat_template(conversation, return_tensors="pt", return_dict=True).to("cuda")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=False)
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens = True).strip()

    # Gemma 3 it
    elif model_path in {"google/gemma-3-1b-it", "google/gemma-3-4b-it"}:

        conversation = [
            [
                {
                    "role": "system", 
                    "content": [{"type": "text", "text": SISTEM_PROMPT},]
                },
                {
                    "role": "user", 
                    "content": [{"type": "text", "text": prompt},]
                },
            ]
        ]

        inputs = tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=True, 
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
            ).to('cuda')

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=False)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    # Llama 3.x base
    elif model_path in {"meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"}:

        inputs = tokenizer(SISTEM_PROMPT + "\n" + prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=False)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens = True).strip().split("\n")[0]

    # Llama 3.x it
    elif model_path in {"meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"}:

        conversation = [
            {
                "role": "system", 
                "content": SISTEM_PROMPT
            },
            {
                "role": "user", 
                "content": prompt
            },
        ]
        
        inputs = tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
            ).to('cuda')
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=False)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
   
    # Qwen 3 it
    elif model_path in {"Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B"}:

        conversation = [
            {
                "role": "system", 
                "content": SISTEM_PROMPT
            },
            {
                "role": "user", 
                "content": prompt
            },
        ]

        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = tokenizer(text, return_tensors="pt").to('cuda')

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=False)
        
        generated_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
        
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        del generated_ids

    # Mistral base
    elif model_path in {"mistralai/Mistral-7B-v0.3"}:
        inputs = tokenizer(SISTEM_PROMPT + "\n" + prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=False)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens = True).strip().split("\n")[0]
    
    # Mistral it
    elif model_path in {"mistralai/Mistral-7B-Instruct-v0.3"}:

        conversation = [
            {
                "role": "system", 
                "content": SISTEM_PROMPT
            },
            {
                "role": "user", 
                "content": prompt
            },
        ]

        inputs = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to('cuda')

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=False)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # Openchat
    elif model_path in {"openchat/openchat-3.5-0106"}:
        
        conversation = [
            {"role": "user", "content": SISTEM_PROMPT + "\n" + prompt},
        ]

        inputs = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=False)
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens = True).strip()

    # Deepseek it
    elif model_path in {"deepseek-ai/deepseek-llm-7b-chat"}:

        model.generation_config = GenerationConfig.from_pretrained(model_path)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        
        conversation = [
            {
                "role": "system", 
                "content": SISTEM_PROMPT
            },
            {
                "role": "user", 
                "content": prompt
            },
        ]

        inputs = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to('cuda')

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=False)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    # Solar base
    elif model_path in {"upstage/SOLAR-10.7B-v1.0"}:
    
        inputs = tokenizer(SISTEM_PROMPT + "\n" + prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=False)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens = True).strip().split("\n")[0]
    
    # Solar it
    elif model_path in {"upstage/SOLAR-10.7B-Instruct-v1.0"}:
        
        conversation = [
            {
                "role": "system", 
                "content": SISTEM_PROMPT
            },
            {
                "role": "user", 
                "content": prompt
            },
        ]

        inputs = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to('cuda')

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=False)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    del inputs
    del outputs


    return response

if __name__ == '__main__':

    print("Begin of zs " + model_name + " " + args.a)

    model = AutoModelForCausalLM.from_pretrained(args.m, device_map="auto")

    if args.m == "openchat/openchat-3.5-0106":
        gen_cfg = model.generation_config or GenerationConfig()
        gen_cfg.temperature = None
        gen_cfg.top_p = None
        gen_cfg.top_k = None
        gen_cfg.do_sample = False
        model.generation_config = gen_cfg
        

    tokenizer = AutoTokenizer.from_pretrained(args.m)
        
    if args.a == "qa":
        prompt_path = "./qa_prompt.txt"
    else:
        prompt_path = "./summ_prompt.txt"

    with open(prompt_path, "r", encoding="utf-8") as f:
        initial_prompt = f.read()

    with open("./results/" + model_name  + "_" + args.a + "_responses.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Título", "Respuesta"])

    responses = []

    for index, row in data.iterrows():

        prompt = initial_prompt.replace("{article_headline}", row["Título"])
        prompt = initial_prompt.replace("{article_body}", row["Cuerpo"])

        response = generate_response(args.m, prompt, model, tokenizer)

        with open("./results/" + model_name  + "_" + args.a + "_responses.csv", mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([row["Título"], response])
        
        responses.append(response.strip())

        torch.cuda.empty_cache()
        gc.collect()
    
    refs = data["Respuesta"].astype(str).tolist()

    rouge = evaluate.load("rouge")
    
    rouge_scores = rouge.compute(
        predictions=responses,
        references=refs,
        use_stemmer=True,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"]
    )

    print(rouge_scores)

    df_rouge = pd.DataFrame([
        {"metric": k, "score": v}
        for k, v in rouge_scores.items()
    ])
        
    df_rouge.to_csv("./results/" + model_name  + "_" + args.a + "_metrics.csv")
    
    print("End of zs " + model_name + " " + args.a)
