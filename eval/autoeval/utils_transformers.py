import multiprocessing as mp
import numpy as np
import os
import torch
import transformers
import tqdm
from transformers import AutoFeatureExtractor, AutoProcessor

import utils_gpt

def load_model(model_cls, model_kwargs):
    model = getattr(transformers, model_cls).from_pretrained(**model_kwargs)
    processor = AutoProcessor.from_pretrained(model_kwargs["pretrained_model_name_or_path"])
    return model, processor

def transformers_create_message(prompt, system_prompt=None, image_key=None, images=None):
    # transformers uses the same format as gpt but feeds the raw image
    return utils_gpt.gpt_create_message(prompt, system_prompt, image_key, images, image_url=False)
    
@torch.inference_mode()
def vlm_call(model, processor, messages, max_new_tokens=512, batch_size=8):
    outputs = []
    for i in tqdm(range(0, len(messages), batch_size)):
        batch_messages = messages[i:i+batch_size]
        inputs = processor.apply_chat_template(
            batch_messages, 
            tokenize=True, 
            add_generation_prompt=True,
            return_dict=True, 
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        outputs.extend(
                processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
        )
    return outputs

def vlm_load_call(local_rank, transformers_kwargs, chunk):
    device = f"cuda:{local_rank}"
    transformers_kwargs["model_kwargs"]["device_map"] = device
    model, processor = load_model(transformers_kwargs["model_cls"], transformers_kwargs["model_kwargs"])
    processor.tokenizer.padding_side = "left"
    model.to(device).eval()
    return vlm_call(model, processor, chunk, **transformers_kwargs["forward_kwargs"])

def _transformers_batch_call(transformers_kwargs, chunk, local_rank, return_dict):
    call_fn = eval(transformers_kwargs.get("call_fn", "vlm_load_call"))
    outputs = call_fn(local_rank, transformers_kwargs, chunk)
    return_dict[local_rank] = outputs
    
def transformers_batch_call(
    transformers_kwargs,
    messages
):
    def _chunk(seq, n):
        k, m = divmod(len(seq), n)
        return [seq[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]
 
    gpu_ids = transformers_kwargs.get("gpu_ids", [])
    if not gpu_ids:
        gpu_ids = list(range(torch.cuda.device_count()))
    n_gpus = min(len(gpu_ids), len(messages))
    chunks = _chunk(messages, n_gpus)
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    return_dict = manager.dict()

    procs = []
    for gpu_id, chunk in zip(gpu_ids, chunks):
        p = ctx.Process(
            target=_transformers_batch_call,
            args=(transformers_kwargs, chunk, gpu_id, return_dict),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    outputs = []
    for gpu_id in gpu_ids[:n_gpus]:
        outputs.extend(return_dict[gpu_id])
    return outputs