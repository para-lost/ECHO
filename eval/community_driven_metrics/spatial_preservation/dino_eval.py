# =================================================================
# This code is adapted from extractor.py in Splice
# Original source: https://github.com/omerbt/Splice/blob/master/models/extractor.py
# =================================================================

from torchvision import transforms as T
import torch
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from sklearn.decomposition import PCA
from torchvision.transforms import Resize
from torchvision import transforms
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from dis import Instruction
import os, re, json, base64, argparse, concurrent.futures
from pathlib import Path
from typing import Dict, Any, Optional
import openai                     
from datasets import load_dataset
from PIL import Image
from io import BytesIO


def attn_cosine_sim(x, eps=1e-08):
    x = x[0]  # TEMP: getting rid of redundant dimension, TBF
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix


class VitExtractor:
    BLOCK_KEY = 'block'
    ATTN_KEY = 'attn'
    PATCH_IMD_KEY = 'patch_imd'
    QKV_KEY = 'qkv'
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name, device):
        self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self._init_hooks_data()

    def _init_hooks_data(self):
        self.layers_dict[VitExtractor.BLOCK_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.ATTN_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.QKV_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for key in VitExtractor.KEY_LIST:
            # self.layers_dict[key] = kwargs[key] if key in kwargs.keys() else []
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self):
        def _get_qkv_output(model, inp, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    # TODO: CHECK ATTN OUTPUT TUPLE
    def _get_patch_imd_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    def get_feature_from_input(self, input_img):  # List([B, N, D])
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_qkv_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_attn_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.ATTN_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_patch_size(self):
        return 8 if "8" in self.model_name else 16

    def get_width_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return w // patch_size

    def get_height_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return h // patch_size

    def get_patch_num(self, input_img_shape):
        patch_num = 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))
        return patch_num

    def get_head_num(self):
        if "dino" in self.model_name:
            return 6 if "s" in self.model_name else 12
        return 6 if "small" in self.model_name else 12

    def get_embedding_dim(self):
        if "dino" in self.model_name:
            return 384 if "s" in self.model_name else 768
        return 384 if "small" in self.model_name else 768

    def get_queries_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        q = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[0]
        return q

    def get_keys_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        k = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[1]
        return k

    def get_values_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        v = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[2]
        return v

    def get_keys_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)
        if len(qkv_features) <= layer_num:
            return None
        print(len(qkv_features))
        qkv_features = qkv_features[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        return keys

    def get_keys_self_sim_from_input(self, input_img, layer_num):
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        if keys is None:
            return None
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_keys[None, None, ...])
        return ssim_map


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


extractor = VitExtractor("dino_vits16", device)

to_rgb = T.Lambda(lambda img: img.convert("RGB"))
imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
global_resize_transform = Resize((224, 224))

global_transform = transforms.Compose([to_rgb,global_resize_transform,
                                        T.ToTensor(),
                                            imagenet_norm
                                            ])
                                                    
def calculate_global_ssim_loss(outputs, inputs):
    loss = 0.0
    a = inputs
    b = outputs
    a = global_transform(a).to(device)
    b = global_transform(b).to(device)
    with torch.no_grad():   
        target_keys_self_sim = extractor.get_keys_self_sim_from_input(a.unsqueeze(0), layer_num=11)
        if target_keys_self_sim is None:
            return 0.0
    keys_ssim = extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
    if keys_ssim is None:
        return 0.0
    loss += F.mse_loss(keys_ssim, target_keys_self_sim).item()
    return loss



def parse_args():
    p = argparse.ArgumentParser("Strict image-quality auditor (OpenAI)")
    p.add_argument("--dataset", default="echo-bench/echo2025-mit",
                   help="HF dataset name or local path")
    p.add_argument("--config",   default="image_to_image",
                   help="Dataset config to evaluate (default: image_to_image)")
    p.add_argument("--split",   default="test",
                   help="Dataset split to evaluate (default: test)")
    p.add_argument("--outdir",  default="./results",
                   help="Where to dump *.jsonl result files")
    p.add_argument("--model",  required=True,
                   help="Which model to evaluate on")
    p.add_argument("--gen-dir", default='../model_results',
                   help="Root foldern that contains the generated images of the baseline models")
    p.add_argument("--max-threads", type=int, default=1) # Set to 1 else the cache gets messed up!
    return p.parse_args()


# ----------------------------------------------------------------------
# 2. Modify evaluate() to compute cosine similarity
# ----------------------------------------------------------------------
def get_model_image_path(root_folder: str, config: str, 
                        model_name: str, id: str) -> str:
    """
    Based on the model name, get the path to the image
    Models: Anole, Bagel_think, Bagel, LLM_DM
    """
    if os.path.exists(f"{root_folder}/{model_name}/{config}/{id}/gen_0.png"):
        return f"{root_folder}/{model_name}/{config}/{id}/gen_0.png"
    else:
        return None
        
def evaluate(sample_id: int, sample: Dict[str, Any],
             gen_root: str, split: str, model_name: str) -> Optional[Dict[str, Any]]:
    
    # Get input images (could be multiple)
    input_imgs = sample["input_images"]
    if not isinstance(input_imgs, list):
        input_imgs = [input_imgs]
    text_only = False
    
    # Locate generated image
    gen_path = get_model_image_path(gen_root, split, model_name, sample_id, text_only)
    if not gen_path or not os.path.exists(gen_path):
        print(f"[!] missing {gen_path}")
        return {
            "id": sample_id,
            "prompt": sample["prompt"],
            "similarity": 0.0,
            "text_only": False
        }
    
    output_img = Image.open(gen_path).convert("RGB")
    
    # Compute max cosine similarity across all input images
    sims = []
    for in_img in input_imgs:
        sim = calculate_global_ssim_loss(output_img, in_img)
        sims.append(sim)
    
    max_sim = max(sims) if sims else 0.0
    
    return {
        "id": sample_id,
        "prompt": sample["prompt"],
        "similarity": max_sim,
        "text_only": False
    }

if __name__ == "__main__":
    args = parse_args()

    # 1. Load dataset
    ds = load_dataset(args.dataset, name=args.config, split=args.split, trust_remote_code=True)
    with open("../../image_to_image_assigment.json", "r") as f:
        metric_assignments = json.load(f)

    idx_list = [s["id"] for s in ds if metric_assignments[str(s["id"])]["Spatial Position Preservation"] == 1]
    ds = [s for s in ds if metric_assignments[str(s["id"])]["Spatial Position Preservation"] == 1]

    print(f"Dataset loaded: {args.dataset}[{args.config}][{args.split}] – {len(ds):,} samples")

    # 2. Prepare output paths
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    full_path   = outdir / "eval_full.jsonl"
    sims_path   = outdir / "eval_sims.jsonl"

    # 3. Skip already processed IDs (resume support)
    done = set()
    if full_path.exists():
        with open(full_path, "r") as f:
            done = {json.loads(ln)["id"] for ln in f}
        print(f"Resuming – {len(done)} samples already finished")

    results = []

    # 4. Multithreaded DINO eval
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_threads) as pool, \
        full_path.open("a", encoding="utf-8") as fout_full, \
        sims_path.open("a", encoding="utf-8") as fout_sims:

        # instead of using the full ds, we only want the samples that needs dino eval, specifically, we need to read the ../get_metrics_assigment/assignments/metric_assignments_{args.config}.json and get those whose "Spatial Position Preservation" is 1
        futures = {
            pool.submit(evaluate, i, s, args.gen_dir, args.config, args.model): i
            for i, s in zip(idx_list, ds) if i not in done
        }

        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            fout_full.write(json.dumps(res, ensure_ascii=False) + "\n")
            fout_sims.write(json.dumps({"id": res["id"], "similarity": res["similarity"]}) + "\n")
            results.append(res)
            print(f"[✓] id={res['id']}  Sim={res['similarity']:.4f}")

    # 5. Compute averages directly from the saved file
    with open(full_path, "r") as f:
        # only get the data for the idx_list
        data = [json.loads(ln) for ln in f if int(json.loads(ln)["id"]) in idx_list]

    non_text_only = [r for r in data if not r["text_only"]]
    # text_only     = [r for r in data if r["text_only"]]
    
    # only compute for those similarity is not 0
    non_text_only = [r for r in non_text_only if r["similarity"] != 0]
    if non_text_only:
        avg_non_text = sum(r["similarity"] for r in non_text_only) / len(non_text_only)
        print(f"Average similarity (non-text-only): {avg_non_text:.4f}")
    else:
        print("No non-text-only samples")
