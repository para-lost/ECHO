"""
Download Echo-Bench image_to_image split and materialize images + crops.

Requirements:
  pip install datasets pillow requests tqdm

What it does:
  data/images_to_image/
    <id>/
      input_images/
        input_1.png, input_2.png, ...
      crops/   (only if is_screenshot and bboxes exist)
        input_1_crop_1.png, input_1_crop_2.png, ...
"""

import os
import json
import ast
import time
import math
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import requests
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

# ------------------ CONFIG ------------------
REPO_ID = "echo-bench/echo2025"    # dataset repo id
CONFIG  = "image_to_image"           # which config to pull
SPLIT   = "test"                    # split name
OUT_DIR = Path("data/images_to_image")
# Build twimg URLs from media IDs; you can change name= to "large", "orig", etc.
TWIMG_QUERY = "?format=jpg"
# --------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_jsonish(cell: Any):
    """Parse JSON/Python-repr lists/dicts from a cell. Return raw value if not parseable."""
    if cell is None:
        return None
    if isinstance(cell, (list, dict)):
        return cell
    s = str(cell).strip()
    if not s:
        return None
    # unwrap outer quotes & collapse doubled quotes (common in CSV)
    if (s[0] in "\"'") and (s[-1] == s[0]):
        s = s[1:-1]
    s = s.replace('""', '"')
    # try json
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return None
    return None

def ensure_list_of_str(cell: Any) -> List[str]:
    """Turn a cell into list[str]. Accept list, JSON string, single string."""
    if isinstance(cell, list):
        return [str(x) for x in cell]
    parsed = parse_jsonish(cell)
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    if isinstance(cell, str) and cell:
        return [cell]
    return []

def ensure_boxes(cell: Any) -> List[List[int]]:
    """
    Normalize bboxes: [[x1,y1,x2,y2], ...] as ints.
    Accepts list, JSON string, python-repr; ignores malformed boxes.
    """
    arr = None
    if isinstance(cell, list):
        arr = cell
    else:
        parsed = parse_jsonish(cell)
        if isinstance(parsed, list):
            arr = parsed
    if arr is None:
        return []
    out: List[List[int]] = []
    for b in arr:
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                out.append([int(round(float(v))) for v in b])
            except Exception:
                pass
    return out

# Simple check for "already a URL"
_URL_RE = re.compile(r"^https?://", re.I)

def media_id_to_url(token: str) -> str:
    """
    Build a download URL from a media token.
    If it's already a URL, return as-is.
    Otherwise assume Twitter media id.
    """
    if _URL_RE.match(token or ""):
        return token
    # Default to Twitter media: https://pbs.twimg.com/media/<ID>?format=jpg&name=...
    return f"https://pbs.twimg.com/media/{token}{TWIMG_QUERY}"

def safe_download(url: str, dst: Path, retries: int = 3, timeout: int = 20) -> bool:
    """Download with retries. Returns True on success."""
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"status {r.status_code}")
                tmp = dst.with_suffix(dst.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dst)
                return True
        except Exception as e:
            if attempt == retries:
                return False
            time.sleep(1.5 * attempt)
    return False

def ensure_png(infile: Path, outfile: Path) -> bool:
    """Convert any raster to PNG (or just re-save). Returns True on success."""
    try:
        with Image.open(infile) as im:
            im = im.convert("RGB")
            outfile.parent.mkdir(parents=True, exist_ok=True)
            im.save(outfile, format="PNG")
        return True
    except Exception:
        return False

def crop_with_bboxes(img_path: Path, bboxes: List[List[int]], out_dir: Path, prefix: str):
    """Save crops as <prefix>_crop_<k>.png for each bbox."""
    if not bboxes:
        return
    try:
        with Image.open(img_path) as im:
            W, H = im.size
            out_dir.mkdir(parents=True, exist_ok=True)
            for k, (x1, y1, x2, y2) in enumerate(bboxes, 1):
                # clamp to image bounds
                x1c = max(0, min(int(x1), W - 1))
                y1c = max(0, min(int(y1), H - 1))
                x2c = max(0, min(int(x2), W))
                y2c = max(0, min(int(y2), H))
                if x2c <= x1c or y2c <= y1c:
                    continue
                crop = im.crop((x1c, y1c, x2c, y2c))
                crop = crop.convert("RGB")
                crop.save(out_dir / f"{prefix}_crop_{k}.png")
    except Exception:
        pass

def main():
    # 1) load the split
    ds = load_dataset(REPO_ID, CONFIG, split=SPLIT)
    print(f"Loaded: {REPO_ID} [{CONFIG}/{SPLIT}] — {len(ds)} rows")

    # Column names used in your image_to_image config
    COL_ID           = "id"
    COL_INPUTS       = "input_images"     # list[str] media tokens or URLs
    COL_IS_SCREEN    = "is_screenshot"    # bool
    COL_INPUT_BOXXES = "input_bboxs"      # list[list[int]] or JSON string

    # 2) iterate rows
    for row in tqdm(ds, desc="Downloading & cropping"):
        rid = str(row.get(COL_ID, "") or "").strip()
        if not rid:
            continue

        # prepare dirs
        id_dir = OUT_DIR / rid
        inputs_dir = id_dir / "input_images"
        crops_dir  = id_dir / "crops"
        inputs_dir.mkdir(parents=True, exist_ok=True)

        # inputs: list of media tokens/URLs
        tokens = row.get(COL_INPUTS, []) or []
        if not isinstance(tokens, list):
            tokens = [tokens]

        # bboxes (may be JSON string or list). If multiple images are present and
        # you have one bbox list per image in future, you can adapt mapping here.
        is_screenshot = bool(row.get(COL_IS_SCREEN, False))
        input_boxes = ensure_boxes(row.get(COL_INPUT_BOXXES))

        # 3) download + convert to PNG
        for i, token in enumerate(tokens, start=1):
            url = media_id_to_url(str(token))
            tmp_jpg  = inputs_dir / f"input_{i}.jpg"
            final_png = inputs_dir / f"input_{i}.png"
            # check if the file exists

            ok = safe_download(url, tmp_jpg)
            if not ok:
                # fallback: try without query (some IDs only work with ?name=large or ?name=orig)
                alt_url = url.split("?", 1)[0]
                ok = safe_download(alt_url, tmp_jpg)
                if not ok:
                    print(f"[skip] {rid} input_{i}: failed to download {url}")
                    continue

            if not ensure_png(tmp_jpg, final_png):
                print(f"[warn] {rid} input_{i}: failed to convert to PNG")
                continue
            try:
                tmp_jpg.unlink(missing_ok=True)
            except Exception:
                pass

            # 4) optional crops when screenshot
            if is_screenshot and input_boxes:
                crop_with_bboxes(final_png, input_boxes, crops_dir, prefix=f"input_{i}")

    print("Done. Output root:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
