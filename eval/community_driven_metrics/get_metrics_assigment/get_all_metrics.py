#!/usr/bin/env python3
"""
This code iterates through the samples in ECHO and assigns the metrics needed for each sample.
Uses GPT to determine which metrics are applicable for each sample based on the classify.yaml template.
"""

import os
import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import openai
from datasets import load_dataset
from PIL import Image
import base64
from io import BytesIO
import concurrent.futures
from tqdm import tqdm
import glob
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Assign metrics to ECHO samples using GPT")
    parser.add_argument("--dataset", default="echo-bench/echo2025-mit",
                       help="HuggingFace dataset name or local path")
    parser.add_argument("--config", default="image_to_image",
                       help="Dataset config to process (default: image_to_image)")
    parser.add_argument("--split", default="test",
                       help="Dataset split to process (default: test)")
    parser.add_argument("--outdir", required=True,
                       help="Output directory for results")
    parser.add_argument("--max-threads", type=int, default=10,
                       help="Maximum number of concurrent threads")
    parser.add_argument("--start-idx", type=int, default=0,
                       help="Starting index for processing")
    parser.add_argument("--end-idx", type=int, default=None,
                       help="Ending index for processing (None for all)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing results")
    return parser.parse_args()

def load_metrics_config(config_path: str) -> Dict[str, Any]:
    """Load the metrics classification configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_metric_lists(metrics_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    """Load metric lists from the metrics directories with full metadata."""
    metric_lists = {}
    
    # Load image_to_image metrics
    interleaved_dir = metrics_dir / "image_to_image_metrics"
    if interleaved_dir.exists():
        interleaved_metrics = []
        for yaml_file in sorted(interleaved_dir.glob("*.yaml")):
            with open(yaml_file, 'r') as f:
                metric_data = yaml.safe_load(f)
                if 'name' in metric_data:
                    metric_info = {
                        'name': metric_data['name'],
                        'description': metric_data.get('description', ''),
                        'applicability': metric_data.get('applicability', '')
                    }
                    interleaved_metrics.append(metric_info)
        metric_lists['image_to_image'] = interleaved_metrics
        print(f"Loaded {len(interleaved_metrics)} image_to_image metrics")
    
    # Load text-only metrics
    text_only_dir = metrics_dir / "text_to_image_metrics"
    if text_only_dir.exists():
        text_only_metrics = []
        for yaml_file in sorted(text_only_dir.glob("*.yaml")):
            with open(yaml_file, 'r') as f:
                metric_data = yaml.safe_load(f)
                if 'name' in metric_data:
                    metric_info = {
                        'name': metric_data['name'],
                        'description': metric_data.get('description', ''),
                        'applicability': metric_data.get('applicability', '')
                    }
                    text_only_metrics.append(metric_info)
        metric_lists['text_to_image'] = text_only_metrics
        print(f"Loaded {len(text_only_metrics)} text-only metrics")
    
    return metric_lists

def pil_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def is_text_only(sample: Dict[str, Any]) -> bool:
    """Check if sample is text-only (no input images)."""
    return (sample.get("input_images") is None) or (len(sample.get("input_images", [])) == 0)

def get_input_images(sample: Dict[str, Any]) -> List[str]:
    """Get the input images for a sample."""

    input_images = sample.get("input_images", [])
    if isinstance(input_images[0], str):
        if sample["is_screenshot"]:
            # glob all the .png files under ../../data/images_to_image/sample["id"]/crops/
            return [Image.open(f) for f in glob.glob(f"../../data/images_to_image/{sample['id']}/crops/*.png")]
        else:
            return [Image.open(f) for f in glob.glob(f"../../data/images_to_image/{sample['id']}/input_images/*.png")]
    else:
        return input_images


def build_classification_prompt(sample: Dict[str, Any], metric_list: List[Dict[str, str]], config: Dict[str, Any]) -> str:
    """Build the prompt for metric classification."""
    task = sample.get("task", "image_generation")
    input_prompt = sample.get("prompt", "")
    
    # Handle input images
    input_images_info = ""
    if not is_text_only(sample):
        input_images = sample.get("input_images", [])
        input_images_info = f"Input images: {len(input_images)} image(s) provided"
    else:
        input_images_info = "Input images: None (text-only generation)"
    
    # Build the prompt using the template from classify.yaml
    prompt_template = config["prompt"]
    
    # Create detailed metric list with descriptions
    metric_details = []
    for metric in metric_list:
        metric_details.append(f"- {metric['name']}: \n Description: {metric['description']} \n Applicability: {metric['applicability']}")
    
    # Replace placeholders
    prompt = prompt_template.replace("<metric_list>", "\n".join(metric_details))
    prompt = prompt.replace("<task>", task)
    prompt = prompt.replace("<input_prompt>", input_prompt)
    prompt = prompt.replace("<input_images>", input_images_info)
    
    return prompt

def classify_metrics(sample: Dict[str, Any], metric_list: List[Dict[str, str]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Use GPT to classify which metrics are applicable for a sample."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Build the classification prompt
    prompt = build_classification_prompt(sample, metric_list, config)
    
    # Prepare images if any
    messages = []
    if not is_text_only(sample):
        # Add input images to the message
        input_images = get_input_images(sample)
        image_b64_list = []
        for img in input_images:
            if isinstance(img, Image.Image):
                image_b64_list.append(pil_to_base64(img))
            else:
                # Assume it's already a PIL image or can be converted
                image_b64_list.append(pil_to_base64(Image.open(img)))
        
        # Create message with images
        content = [{"type": "text", "text": prompt}]
        for img_b64 in image_b64_list:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })
        
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": prompt}]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=1000,
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            # Extract JSON from the response
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = result_text[json_start:json_end]
                result = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response: {result_text}")
            # Return default result with all metrics as 0
            result = {metric['name']: 0 for metric in metric_list}
            result["rationale"] = "Failed to parse GPT response"
            result["prompt"] = sample.get("prompt", "")
        
        return result
        
    except Exception as e:
        print(f"Error calling GPT: {e}")
        # Return default result with all metrics as 0
        result = {metric['name']: 0 for metric in metric_list}
        result["rationale"] = f"GPT call failed: {str(e)}"
        result["prompt"] = sample.get("prompt", "")
        return result

def process_sample(sample_id: int, sample: Dict[str, Any], metric_lists: Dict[str, List[Dict[str, str]]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single sample to assign metrics."""
    print(f"Processing sample {sample_id}...")
    
    # Determine which metric list to use based on sample type
    text_to_image = is_text_only(sample)
    metric_list = metric_lists['text_to_image'] if text_to_image else metric_lists['image_to_image']
    
    # Classify metrics for this sample
    classification_result = classify_metrics(sample, metric_list, config)
    
    # Prepare the result
    result = {
        "id": sample_id,
        "task": "text_to_image" if text_to_image else "image_to_image",
        "prompt": sample.get("prompt", ""),
        "text_to_image": text_to_image,
        "input_images_count": len(sample.get("input_images", [])) if not text_to_image else 0,
        "metric_type": "text_to_image" if text_to_image else "image_to_image",
        "metric_classification": classification_result,
        "applicable_metrics": [metric for metric, applicable in classification_result.items() 
                              if metric not in ["rationale", "prompt"] and applicable == 1],
        "metric_definitions": {metric['name']: {
            'description': metric['description'],
            'applicability': metric['applicability']
        } for metric in metric_list}
    }
    
    return result

def load_existing_results(output_file: Path) -> set:
    """Load existing result IDs to support resume functionality."""
    if not output_file.exists():
        return set()
    
    existing_ids = set()
    with open(output_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                existing_ids.add(data['id'])
            except (json.JSONDecodeError, KeyError):
                continue
    return existing_ids

def create_dict_from_jsonl(jsonl_file: str, output_file: str):
    import json

    # Read the JSONL file and create the dictionary
    result_dict = {}

    # Process each line in the JSONL file
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            result_dict[data['id']] = data['metric_classification']

    # Save the dictionary to a new JSON file
    with open(output_file, 'w') as file:
        json.dump(result_dict, file, indent=2)

    print(f"Dictionary saved to: {output_file}")
    print(f"Total entries processed: {len(result_dict)}")



def main():
    """Main function to process ECHO samples and assign metrics."""
    args = parse_args()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Environment variable OPENAI_API_KEY is not set.")
    
    # Load the metrics configuration
    config_path = Path(__file__).parent / "classify.yaml"
    config = load_metrics_config(config_path)
    
    # Load metric lists from the metrics directories
    metrics_dir = Path(__file__).parent
    metric_lists = load_metric_lists(metrics_dir)
    
    if not metric_lists:
        raise RuntimeError("No metric lists found in the metrics directories")
    
    # Load the dataset
    print(f"Loading dataset: {args.dataset}[{args.split}]")
    ds = load_dataset(args.dataset, name=args.config, split=args.split, trust_remote_code=True)
    print(f"Dataset loaded: {len(ds)} samples")
    
    # Create output directory
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"metric_assignments_{args.config}.jsonl"
    
    # Determine which samples to process
    if args.resume:
        existing_ids = load_existing_results(output_file)
        print(f"Found {len(existing_ids)} existing results, resuming...")
        indices_to_process = [i for i in range(len(ds)) if i not in existing_ids]
    else:
        indices_to_process = list(range(len(ds)))
    
    # Apply start/end index filters
    if args.start_idx > 0:
        indices_to_process = [i for i in indices_to_process if i >= args.start_idx]
    if args.end_idx is not None:
        indices_to_process = [i for i in indices_to_process if i < args.end_idx]
    
    print(f"Processing {len(indices_to_process)} samples...")
    
    # Process samples with threading
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_sample, i, ds[i], metric_lists, config): i 
            for i in indices_to_process
        }
        
        # Process completed tasks
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), 
                          total=len(future_to_idx), desc="Processing samples"):
            try:
                result = future.result()
                results.append(result)
                
                # Write result immediately to file
                with open(output_file, 'a') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                print(f"✓ Sample {result['id']} ({result['metric_type']}): {len(result['applicable_metrics'])} applicable metrics")
                
            except Exception as e:
                idx = future_to_idx[future]
                print(f"✗ Error processing sample {idx}: {e}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    total_samples = len(results)
    text_only_samples = sum(1 for r in results if r['text_to_image'])
    interleaved_samples = total_samples - text_only_samples
    
    print(f"Total samples processed: {total_samples}")
    print(f"Text-only samples: {text_only_samples}")
    print(f"Interleaved samples: {interleaved_samples}")
    
    # Count metric applicability by type
    text_only_metric_counts = {}
    interleaved_metric_counts = {}
    
    for result in results:
        if result['text_to_image']:
            for metric in result['applicable_metrics']:
                text_only_metric_counts[metric] = text_only_metric_counts.get(metric, 0) + 1
        else:
            for metric in result['applicable_metrics']:
                interleaved_metric_counts[metric] = interleaved_metric_counts.get(metric, 0) + 1
    
    print(f"\nText-only metric applicability counts:")
    for metric, count in sorted(text_only_metric_counts.items()):
        percentage = (count / text_only_samples) * 100 if text_only_samples > 0 else 0
        print(f"  {metric}: {count}/{text_only_samples} ({percentage:.1f}%)")
    
    print(f"\nInterleaved metric applicability counts:")
    for metric, count in sorted(interleaved_metric_counts.items()):
        percentage = (count / interleaved_samples) * 100 if interleaved_samples > 0 else 0
        print(f"  {metric}: {count}/{interleaved_samples} ({percentage:.1f}%)")
    
    print(f"\nResults saved to: {output_file}")
    create_dict_from_jsonl(output_file, f"{output_dir}/metric_assignments_{args.config}.json")
    
if __name__ == "__main__":
    main()
