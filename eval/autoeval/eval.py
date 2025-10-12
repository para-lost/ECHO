import re
from collections import defaultdict, Counter
import os
from PIL import Image
from omegaconf import OmegaConf
import subprocess
from datasets import load_dataset
import asyncio
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import itertools
import random
import glob

import sys
# sys.path.append("../")
from utils_transformers import *
from utils_gpt import *
from utils_arena_hard import *
from utils_arena_fastchat import *


import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        default="gpt",
        choices=["gpt", "transformers", "qwen", "gemini"],
        help="Backend to use for evaluation."
    )
    p.add_argument(
        "--root-path",
        required=True,
        help="Root path to save the results."
    )
    return p.parse_args()
    
# ====================
#  Human Correlation
# ====================
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

# =================
#    Arena Hard
# =================
def get_arena_hard_battles(results, model, category=None, weight=3):
    patterns = ["\[\[([AB<>=]+)\]\]", "\[([AB<>=]+)\]"]
    results = {k: utils_arena_hard.get_score(v, patterns) for k, v in results.items()}
    # weight = duplicate rows to upweight opinion
    label_to_score = {
        "A>B": [1],
        "A>>B": [1] * weight,
        "A=B": [0.5],
        "A<<B": [0] * weight,
        "A<B": [0],
        "B>A": [0],
        "B>>A": [0] * weight,
        "B=A": [0.5],
        "B<<A": [1] * weight,
        "B<A": [1],
    }
    results = {k: label_to_score[v] for k, v in results.items()}
    # create df
    df = []
    for k, v in results.items():
        if "output-baseline" in k:
            k_fwd = k
            k_rev = k.replace("output-baseline", "baseline-output")
            scores = np.array(results[k_fwd]) + 1 - np.array(results[k_rev])
            for score in scores:
                df.append({
                    "uid": "_".join(k.split("_")[:-1]),
                    "model": model,
                    "category": category,
                    "scores": score,
                })
        else:
            continue
    df = pd.DataFrame(df)
    return df

def arena_hard_winrate(results_folder, models, id_to_category=None):
    battles = []
    for model in models:
        results = json.load(open(f"{results_folder}/{model}/raw_results.json"))
        df = get_arena_hard_battles(results, model)
        battles.append(df)
    battles = pd.concat(battles)
    if id_to_category is not None:
        battles["category"] = battles["uid"].map(id_to_category)
    categories = battles["category"].unique()
    leaderboards = []
    for category in categories:
        leaderboard = utils_arena_hard.print_leaderboard(battles[battles["category"] == category], category)
        leaderboard["category"] = category
        leaderboards.append(leaderboard)
    leaderboards = pd.concat(leaderboards)
    return leaderboards, battles

# =================
#    Auto Eval
# =================
def extract_score(txt):
    pattern = re.compile(r'\[\[\s*(10|[1-9])\s*\]\]')
    m = pattern.search(str(txt))
    return int(m.group(1)) if m else None

def format_sample_score(sample, idx, output_image_folder, tag_model=False, image_size=512, config_path="configs/auto_eval.yaml"):
    prompt_config = OmegaConf.load(config_path)

    id = f"{sample['id']}"
    output_image_file = glob.glob(f"{output_image_folder}/{id}*")
    if not output_image_file or not os.path.exists(output_image_file[0]):
        print("no output_image_file")
        return

    # Prepare input images
    input_images = sample["input_images"]
    if input_images and type(input_images[0]) is str:
        input_images = [Image.open(f) for f in input_images]
    print("len of input_images", len(input_images))
    
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
            # print("message", message)
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


if __name__ == "__main__":
    args = parse_args()
    mode = args.mode
    root_path = args.root_path
    config = OmegaConf.load("configs/config.yaml")

    for config in ["text_to_image", "image_to_image_synthetic"]:
        output_image_folder = f"{root_path}/{config}"
        input_ds = load_dataset("echo-bench/echo-bench", name=config, split="test", streaming=False)
        for model in sorted([os.path.basename(f) for f in glob.glob(f"{output_image_folder}/*")]):
            output_file_path=f"./runs/{mode}/{config}/{model}.json"
            if os.path.exists(output_file_path):
                continue
            print(f"Evaluating {model}")
            asyncio.run(experiment_evaluate(
                model_kwargs=config[f"{mode}_kwargs"],
                input_ds=input_ds,
                mode=mode,
                format_sample_fn=format_sample_score,
                output_file=output_file_path,
                format_sample_kwargs={
                    "output_image_folder": f"{output_image_folder}/{model}",
                },
            ))
