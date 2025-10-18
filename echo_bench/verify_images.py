#!/usr/bin/env python3
import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import open_clip
import io, os
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
model.eval()

@torch.no_grad()
def embed_pil(img: Image.Image) -> torch.Tensor:
    """L2-normalized CLIP image embedding."""
    # preprocess returns a CHW tensor normalized as CLIP expects
    x = preprocess(img).unsqueeze(0).to(device)
    feat = model.encode_image(x)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu()

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum().item())

def bytes_to_pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def main():
    parser = argparse.ArgumentParser(description="Verify downloaded images against reference embeddings")
    parser.add_argument("--folder", type=Path, required=True, help="Root folder containing downloaded images")
    parser.add_argument("--embeddings", type=Path, required=True, help="JSON file with reference embeddings")
    parser.add_argument("--threshold", type=float, default=0.90, help="Cosine similarity threshold (default: 0.95)")
    args = parser.parse_args()
    
    # Load reference embeddings
    print(f"Loading reference embeddings from {args.embeddings}...")
    with open(args.embeddings, 'r') as f:
        reference_embeddings = json.load(f)
    
    # Verify images
    checked = 0
    same = 0
    diff = 0
    missing = 0
    
    print(f"Verifying images in {args.folder}...")
    
    for sample_id, sample_embeddings in reference_embeddings.items():
        sample_folder = args.folder / sample_id / "crops"
        if not sample_folder.exists():
            sample_folder = args.folder / sample_id / "input_images"
        
        if not sample_folder.exists():
            print(f"Missing folder: {sample_folder}")
            missing += len(sample_embeddings)
            continue
            
        for idx, ref_embedding in sample_embeddings.items():
            if "crops" in str(sample_folder):
                image_path = sample_folder / f"input_1_crop_{idx}.png"
            else:
                image_path = sample_folder / f"input_{idx}.png"
            
            if not image_path.exists():
                print(f"Missing image: {image_path}")
                missing += 1
                continue
                
            try:
                # Load and embed local image
                local_pil = Image.open(image_path)
                local_emb = embed_pil(local_pil)
                
                # Compare with reference
                ref_emb = np.array(ref_embedding)
                similarity = cosine_sim(local_emb, ref_emb)
                
                checked += 1
                
                if similarity >= args.threshold:
                    same += 1
                else:
                    diff += 1
                    print("--------------------------------")
                    print(f"Sample ID: {sample_id}, Image: {idx}")
                    print(f"Cosine similarity: {similarity:.6f} (threshold {args.threshold})")
                    print(f"Local file: {image_path}")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                missing += 1
    
    print(f"\\nVerification complete!")
    print(f"Checked: {checked} | Same: {same} | Diff: {diff} | Missing: {missing}")
    
    # Print detailed verification results
    print(f"\n{'='*60}")
    print("DETAILED VERIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"Total images processed: {checked + missing}")
    print(f"Successfully verified: {same} ({same/(checked+missing)*100:.1f}% of total)" if (checked+missing) > 0 else "Successfully verified: 0")
    print(f"Failed verification: {diff} ({diff/(checked+missing)*100:.1f}% of total)" if (checked+missing) > 0 else "Failed verification: 0")
    print(f"Missing/Error: {missing} ({missing/(checked+missing)*100:.1f}% of total)" if (checked+missing) > 0 else "Missing/Error: 0")
    print(f"Similarity threshold: {args.threshold}")
    
    if checked > 0:
        success_rate = same / checked * 100
        print(f"Success rate (of checked images): {success_rate:.1f}%")
    


if __name__ == "__main__":
    main()