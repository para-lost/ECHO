import asyncio
from collections import defaultdict, Counter
from datasets import load_dataset
import glob
import itertools
import json
import numpy as np
import os
import pandas as pd
from PIL import Image
import random
import re
import subprocess
import sys
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import utils_transformers
import utils_gpt
import utils_arena_fastchat

import sys
sys.path.append("../../echo_bench")
from echo_bench_i2i import echo_bench_i2i_collate

def extract_score(txt):
    pattern = re.compile(r'\[\[\s*(10|[1-9])\s*\]\]')
    m = pattern.search(str(txt))
    return int(m.group(1)) if m else None

def scores_to_pairwise(results):
    models = results["model"].unique()
    df = []
    for question_id in results["question_id"].unique():
        question_group = results[results["question_id"] == question_id]
        question_model_to_score = {model: float(score) for model, score in zip(question_group["model"], question_group["score"])}
        for model_a, model_b in itertools.combinations(models, 2):
            if model_a not in question_model_to_score or model_b not in question_model_to_score:
                continue
            score_a = question_model_to_score[model_a]
            score_b = question_model_to_score[model_b]
            winner = "model_a" if score_a > score_b else ("model_b" if score_b > score_a else "tie")
            df.append({
                "question_id": question_id,
                "model_a": model_a,
                "model_b": model_b,
                "winner": winner,
                "score_a": score_a,
                "score_b": score_b,
                "margin": abs(score_a - score_b),
            })
    return pd.DataFrame(df)

def format_sample_score(sample, idx, output_image_folder, tag_model=False, image_size=512, config_path="configs/auto_eval.yaml"):
    prompt_config = OmegaConf.load(config_path)

    id = f"{sample['id']}"
    output_image_file = glob.glob(f"{output_image_folder}/{id}*")
    if not output_image_file or not os.path.exists(output_image_file[0]):
        return

    # Prepare input images
    input_images = sample["input_images"]
    if input_images and type(input_images[0]) is str:
        input_images = [Image.open(f) for f in input_images]
    
    # Prepare output image
    output_image = Image.open(output_image_file[0])
    output_images = [output_image]  # Convert to list for consistency
    
    system_prompt = prompt_config.get("system_prompt")
    eval_prompt = str(prompt_config["prompt"])
    eval_prompt = eval_prompt.replace("<input_prompt>", str(sample["prompt"]))
    
    create_message_kwargs = {
        "prompt": eval_prompt, 
        "system_prompt": system_prompt,
        "input_images": input_images,
        "output_images": output_images,
        "image_size": image_size
    }
    all_create_message_kwargs = [create_message_kwargs]
    if tag_model:
        all_ids = [f"{id}_{os.path.basename(output_image_folder)}"]
    else:
        all_ids = [id]
    return all_ids, all_create_message_kwargs

async def experiment_evaluate(model_kwargs, input_ds, format_sample_fn, format_sample_kwargs={}, mode="gpt", output_file=None):
    batch_ids, batch_messages = [], []
    for idx, sample in tqdm(enumerate(input_ds)):
        _format_sample_kwargs = {**format_sample_kwargs, **sample.get("format_sample_kwargs", {})}
        format_sample_result = format_sample_fn(sample, idx, **_format_sample_kwargs)
        if format_sample_result is None:
            continue
        all_ids, all_create_message_kwargs = format_sample_result
        create_message_fn = utils_gpt.qwen_create_message
        for id, create_message_kwargs in zip(all_ids, all_create_message_kwargs):
            message = create_message_fn(**create_message_kwargs)
            batch_messages.append(message)
            batch_ids.append(id)

    if mode == "transformers":
        batch_outputs = utils_transformers.transformers_batch_call(model_kwargs, batch_messages)
    elif mode == "gpt":
        batch_outputs = await utils_gpt.gpt_batch_call(batch_messages, **model_kwargs)
    elif mode == "qwen":
        batch_outputs = await utils_gpt.qwen_batch_call(batch_messages, **model_kwargs)
    elif mode == "gemini":
        batch_outputs = await utils_gpt.gemini_batch_call(batch_messages, **model_kwargs)
    else:
        raise NotImplementedError

    raw_results = {id: output for id, output in zip(batch_ids, batch_outputs)}
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok="True")
        json.dump(raw_results, open(output_file, "w"))
    
    return raw_results

def main(config):
    root_path, mode = config.root_path, config.mode
    for split_name in config.split_names:
        output_image_folder = f"{root_path}/{split_name}"
        input_ds = load_dataset("echo-bench/echo2025", name=split_name, split="test", streaming=False)
        if split_name == "image_to_image":
            assert NotImplementedError("The image-to-image split is not supported yet.")
        clean_results = []
        for model in sorted([os.path.basename(f) for f in glob.glob(f"{output_image_folder}/*")]):
            print(f"Evaluating {model}")
            raw_results = asyncio.run(experiment_evaluate(
                model_kwargs=config[f"{mode}_kwargs"],
                input_ds=input_ds,
                mode=mode,
                format_sample_fn=format_sample_score,
                output_file=f"runs/{split_name}/{mode}/{model}.json",
                format_sample_kwargs={
                    "output_image_folder": f"{output_image_folder}/{model}",
                },
            ))
            for k, v in raw_results.items():
                try:
                    clean_results.append({"question_id": k, "model": model, "score": extract_score(v)})
                except:
                    print(f"Error extracting score for {k}")
        if clean_results:
            pd.DataFrame(clean_results).to_csv(f"runs/{split_name}/{mode}.csv", index=False)

if __name__ == "__main__":
    config = OmegaConf.load("configs/config_eval.yaml")
    config = OmegaConf.merge(config, OmegaConf.from_cli())
    main(config)