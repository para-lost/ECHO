#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence
from pathlib import Path

from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset


class EchoBenchHFImageToImage(Dataset):
    """
    Map-style PyTorch Dataset that wraps the HF image_to_image split.
    Returns each sample as:
      {
        "id": str,
        "prompt": str,
        "input_images": List[PIL.Image.Image],
      }
    """

    def __init__(
        self,
        repo_id: str = "echo-bench/echo2025",
        name: str = "image_to_image",
        split: str = "test",
    ) -> None:
        self.repo_id = repo_id
        self.name = name
        self.split = split

        self.ds = load_dataset(self.repo_id, name=self.name, split=self.split)
        self._indices = list(range(len(self.ds)))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.ds[self._indices[idx]]

        rid = str(ex.get("id"))
        
        prompt = ex.get("prompt") or ""
        path = Path(f"./data/images_to_image/{rid}")
        # see the crops folder
        crops_path = path / "crops"
        inputs_path = path / "input_images"
        if crops_path.exists():
            imgs = [Image.open(crop_path) for crop_path in crops_path.iterdir()]
        else:
            imgs = [Image.open(input_path) for input_path in inputs_path.iterdir()]
       
        return {
            "id": rid,
            "prompt": prompt,
            "input_images": imgs
        }

# ---------- Optional: a simple collate that keeps lists variable-length ----------
def echo_bench_i2i_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "id": [b["id"] for b in batch],
        "prompt": [b["prompt"] for b in batch],
        "input_images": [b["input_images"] for b in batch]
    }
