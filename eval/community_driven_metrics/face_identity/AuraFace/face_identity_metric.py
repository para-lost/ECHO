import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from typing import List, Dict, Tuple, Optional, Union, Any
import json
import argparse
from pathlib import Path
import concurrent.futures
from datasets import load_dataset
from PIL import Image
from typing import Any

# ----------------------------------------------------------------------
# 1. CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Face Identity Metric - AuraFace")
    p.add_argument("--dataset", default="echo-bench/echo2025-mit",
                   help="HF dataset name or local path")
    p.add_argument("--config",   default="image_to_image",
                   help="Dataset config to evaluate (default: image_to_image)")
    p.add_argument("--split",   default="test",
                   help="Dataset split to evaluate (default: test)")
    p.add_argument("--outdir",  default="results/auraface",
                   help="Where to dump result files")
    p.add_argument("--model",  required=True,
                   help="Which model to evaluate on")
    p.add_argument("--modelpath", default='./',
                   help='Path to AuraFace model')
    p.add_argument("--gen-dir", default='../model_results',
                   help="Root foldern that contains the generated images of the baseline models")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Similarity threshold for face identity")
    p.add_argument("--max-threads", type=int, default=30)
    return p.parse_args()
                                                                
# ----------------------------------------------------------------------
# 2. Helper utilities
# ----------------------------------------------------------------------
def validate_image_array(img_array: np.ndarray, img_id: str = "unknown") -> bool:
    """Validate that an image array has valid dimensions and data."""
    if img_array is None:
        print(f"Image array is None for {img_id}")
        return False
    
    if len(img_array.shape) != 3:
        print(f"Invalid image shape {img_array.shape} for {img_id} - expected 3D array")
        return False
    
    height, width, channels = img_array.shape
    if height <= 0 or width <= 0:
        print(f"Invalid image dimensions {height}x{width} for {img_id}")
        return False
    
    if channels not in [1, 3, 4]:
        print(f"Invalid number of channels {channels} for {img_id}")
        return False
    
    if img_array.dtype not in [np.uint8, np.float32, np.float64]:
        print(f"Invalid image dtype {img_array.dtype} for {img_id}")
        return False
    
    # Check for reasonable size limits
    if height > 10000 or width > 10000:
        print(f"Image too large {height}x{width} for {img_id}")
        return False
    
    if height < 10 or width < 10:
        print(f"Image too small {height}x{width} for {img_id}")
        return False
    
    return True

def convert_pil_to_numpy(pil_image: Image.Image | np.ndarray, img_id: str = "unknown") -> Optional[np.ndarray]:
    """Accept PIL.Image or np.ndarray (HWC or 1xHWC). Return uint8 RGB."""
    try:
        if pil_image is None:
            print(f"PIL image is None for {img_id}")
            return None

        # If already ndarray, copy path; else convert from PIL
        if isinstance(pil_image, np.ndarray):
            arr = pil_image
        else:
            arr = np.array(pil_image)

        # Squeeze a leading batch dimension of 1: (1, H, W, C) -> (H, W, C)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]

        # If grayscale (H, W), expand to RGB
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=2)

        # If (H, W, 1), repeat to RGB
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)

        # If RGBA, drop alpha
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]

        # Validate dimensions now
        if not validate_image_array(arr, img_id):
            return None

        # Ensure uint8 [0,255]
        if arr.dtype != np.uint8:
            if arr.dtype in (np.float32, np.float64) and arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        return arr

    except Exception as e:
        print(f"Error converting PIL image to numpy for {img_id}: {e}")
        return None


def get_model_image_path(root_folder: str, config: str, 
                        model_name: str, id: str) -> str:
    """
    Based on the model name, get the path to the image
    Models: Anole, Bagel_think, Bagel, LLM_DM, BLIP3o
    """
    if os.path.exists(f"{root_folder}/{model_name}/{config}/{id}/gen_0.png"):
        return f"{root_folder}/{model_name}/{config}/{id}/gen_0.png"
    else:
        return None
        
def is_text_only(sample: Dict[str, Any]) -> bool:
    return sample["input_images"] is None

# ----------------------------------------------------------------------
# 3. Core Face Identity Metric
# ----------------------------------------------------------------------
class FaceIdentityMetric:
    def __init__(self, model_root: str = "/home/annelee/"):
        self.model_root = model_root
        self.face_app = None
        self._initialize_model()
    
    def _initialize_model(self):
        try:
            self.face_app = FaceAnalysis(
                name="auraface",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                root=self.model_root,
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            
        except Exception as e:
            raise RuntimeError(f"Could not initialize AuraFace model: {e}")
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        if not os.path.exists(image_path):
            print(f"Image file '{image_path}' not found.")
            return None
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image '{image_path}' with OpenCV.")
            return None
        
        # Validate loaded image
        if not validate_image_array(img, image_path):
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    
    def extract_face_embedding(self, image: np.ndarray) -> List[Optional[np.ndarray]]:
        try:
            # Validate input image before processing
            if not validate_image_array(image, "face_detection_input"):
                return None
            
            faces = self.face_app.get(image)
            
            if not faces:
                print(" No faces detected in image.")
                return None
            if len(faces) > 1:
                print(f"There are multiple faces detected")
            embeddings = []
            for face in faces:
                embedding = face.embedding
                embeddings.append(embedding)
            return embeddings
            # face = faces[0] # here lets just get the first one
            # embedding = face.embedding
            # return embedding
            
        except Exception as e:
            print(f"Error extracting face embedding: {e}")
            return None
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two face embeddings."""

        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def compute_face_identity_score(self, input_images: List[Union[str, np.ndarray]], output_images: List[Union[str, np.ndarray]], threshold: float = 0.5) -> Dict[str, Union[float, List[float], str]]:
        
        if not input_images:
            raise ValueError("At least one input image must be provided")
        if not output_images:
            raise ValueError("At least one output image must be provided")
        
        input_embeddings = []
        for i, img_input in enumerate(input_images):
            if isinstance(img_input, str):
                img = self.load_image(img_input)
                img_id = img_input
            else:
                img = img_input
                img_id = f"input_{i}"
            
            if img is not None:
                embeddings = self.extract_face_embedding(img)
                if embeddings is not None:
                    for embedding in embeddings:
                        input_embeddings.append((img_id, embedding))
                else:
                    print(f"Could not extract face from input image: {img_id}")
            else:
                print(f"Could not load input image: {img_id}")
        
        if not input_embeddings:
            raise RuntimeError("No valid face embeddings could be extracted from input images")
        
        output_embeddings = []
        for i, img_output in enumerate(output_images):
            if isinstance(img_output, str):
                img = self.load_image(img_output)
                img_id = img_output
            else:
                img = img_output
                img_id = f"output_{i}"
            
            if img is not None:
                embeddings = self.extract_face_embedding(img)
                if embeddings is not None:
                    for embedding in embeddings:
                        output_embeddings.append((img_id, embedding))
                else:
                    print(f"Warning: Could not extract face from output image: {img_id}")
            else:
                print(f"Warning: Could not load output image: {img_id}")
        
        if not output_embeddings:
            raise RuntimeError("No valid face embeddings could be extracted from output images")
        
        # Calculate similarity scores for all input-output pairs
        individual_scores = []
        max_score = 0.0
        
        for input_id, input_emb in input_embeddings:
            for output_id, output_emb in output_embeddings:
                similarity = self.calculate_similarity(input_emb, output_emb)
                individual_scores.append({
                    'input_image': input_id,
                    'output_image': output_id,
                    'similarity': similarity
                })
                max_score = max(max_score, similarity)
        
        result = "SAME_PERSON" if max_score >= threshold else "DIFFERENT_PEOPLE"
        
        return {
            'score': max_score,
            'individual_scores': individual_scores,
            'result': result,
            'threshold': threshold,
            'input_count': len(input_embeddings),
            'output_count': len(output_embeddings),
            'total_comparisons': len(individual_scores)
        }

# ----------------------------------------------------------------------
# 3-2. Core evaluator
# ----------------------------------------------------------------------
def evaluate(sample_id: str, sample: Dict[str, Any],
             gen_root: str, config: str, model_name: str,
             metric: FaceIdentityMetric, threshold: float) -> Optional[Dict[str, Any]]:
    text_only = is_text_only(sample)
    
    if text_only:
        # text-only cannot compute face identity
        return {
            "id": sample_id,
            "prompt": sample["prompt"],
            "evaluation": "Text-only sample - no face identity computation possible",
            "text_only": text_only,
            "score": 0.0,
            "result": "N/A",
            "threshold": threshold
        }
    
    # get input and output image paths
    input_image = sample["input_images"]
    if input_image is None:
        return {
            "id": sample_id,
            "prompt": sample["prompt"],
            "evaluation": "No input image available",
            "text_only": text_only,
            "score": 0.0,
            "result": "N/A",
            "threshold": threshold
        }
    
    gen_path = get_model_image_path(gen_root, config, model_name, str(sample_id))
    if not gen_path or not os.path.exists(gen_path):
        print(f"missing {gen_path}")
        return {
            "id": sample_id,
            "prompt": sample["prompt"],
            "evaluation": f"Generated image not found at {gen_path}",
            "text_only": text_only,
            "score": 0.0,
            "result": "N/A",
            "threshold": threshold
        }
    
    try:
        # Convert PIL input image to numpy array with validation
        input_img_array = convert_pil_to_numpy(input_image, f"sample_{sample_id}_input")
        if input_img_array is None:
            return {
                "id": sample_id,
                "prompt": sample["prompt"],
                "evaluation": "Failed to convert input image to valid array",
                "text_only": text_only,
                "score": 0.0,
                "result": "ERROR",
                "threshold": threshold
            }
        
        # Load generated image
        output_img = cv2.imread(gen_path)
        if output_img is None:
            return {
                "id": sample_id,
                "prompt": sample["prompt"],
                "evaluation": f"Could not load generated image from {gen_path}",
                "text_only": text_only,
                "score": 0.0,
                "result": "N/A",
                "threshold": threshold
            }
        
        # Validate output image
        if not validate_image_array(output_img, gen_path):
            return {
                "id": sample_id,
                "prompt": sample["prompt"],
                "evaluation": f"Invalid generated image at {gen_path}",
                "text_only": text_only,
                "score": 0.0,
                "result": "ERROR",
                "threshold": threshold
            }
        
        output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        
        # Compute face identity score
        result = metric.compute_face_identity_score(
            input_images=[input_img_array],
            output_images=[output_img_rgb], 
            threshold=threshold
        )
        
        return {
            "id": sample_id,
            "prompt": sample["prompt"],
            "evaluation": f"Face identity computed successfully. {result['result']}",
            "text_only": text_only,
            "score": result['score'],
            "result": result['result'],
            "threshold": threshold,
            "individual_scores": result['individual_scores']
        }
        
    except Exception as exc:
        print(f"Error evaluating sample {sample_id}: {exc}")
        return {
            "id": sample_id,
            "prompt": sample["prompt"],
            "evaluation": f"Evaluation failed: {exc}",
            "text_only": text_only,
            "score": 0.0,
            "result": "ERROR",
            "threshold": threshold
        }

# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    # 4.1 Get Auraface model ----------------------------------------------------------------
    try:
        metric = FaceIdentityMetric(model_root=args.modelpath)
    except Exception as e:
        print(f"Failed to initialize AuraFace model: {e}")
        exit(1)

    # 4.2 Dataset ----------------------------------------------------------------------
    ds = load_dataset(args.dataset, name=args.config, split=args.split, trust_remote_code=True)
    with open("../../image_to_image_assigment.json", "r") as f:
        metric_assignments = json.load(f)

    idx_list = [s["id"] for s in ds if metric_assignments[str(s["id"])]["Face Identity Preservation"] == 1]
    ds = [s for s in ds if metric_assignments[str(s["id"])]["Face Identity Preservation"] == 1]

    
    print(f"Dataset loaded: {args.dataset}[{args.config}] – {len(ds):,} samples")

    # 4.3 Output directory -------------------------------------------------------------
    # outdir = Path(args.outdir / args.model) 
    outdir = Path(args.outdir) / args.model
    outdir.mkdir(parents=True, exist_ok=True)
    full_path   = outdir / "face_identity_full.jsonl"
    scores_path = outdir / "face_identity_scores.jsonl"

    # 4.4 Skip already processed IDs ---------------------------------------------------
    done = set()
    if full_path.exists():
        with open(full_path, "r") as f:
            done = {json.loads(ln)["id"] for ln in f}
        print(f"Resuming – {len(done)} samples already finished")

    # 4.5 Multithreaded evaluation -----------------------------------------------------
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_threads) as pool, \
         full_path.open("a", encoding="utf-8") as fout_full, \
         scores_path.open("a", encoding="utf-8") as fout_scores:

        futures = {pool.submit(evaluate, i, s, args.gen_dir, args.config, args.model, metric, args.threshold): i
                   for i, s in zip(idx_list, ds) if i not in done}

        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res is None:   # (no image)
                continue
            # write two streams: full + tiny-scores
            fout_full.write(json.dumps(res, ensure_ascii=False) + "\n")
            score_line = {
                k: res[k] for k in ("id", "score", "result", "threshold")}
            fout_scores.write(json.dumps(score_line) + "\n")
            print(f"[✓] id={res['id']}  "
                  f"Score:{res['score']:.4f} Result:{res['result']}")
    
    # 4.6 Calculate the average score of the dataset
    with open(full_path, "r") as f:
        data = [json.loads(ln) for ln in f]
    
    # Calculate the average for the interleaved samples (non-text-only)
    interleaved_data = [item for item in data if not item.get("text_only") and item.get("id") in idx_list and item.get("score", 0) > 0]
    
    if interleaved_data:
        avg_score = sum(item["score"] for item in interleaved_data) / len(interleaved_data)
        print(f"Average face identity score of the interleaved samples: {avg_score:.4f}")
        print(f"Total interleaved samples processed: {len(interleaved_data)}")
    else:
        print("No valid interleaved samples found for face identity evaluation")
    
    # Count text-only samples
    text_only_count = len([item for item in data if item.get("text_only")])
    print(f"Text-only samples (skipped): {text_only_count}")