# Constantly Improving Image Models Need Constantly Improving Benchmarks

### [Project Page](https://echo-bench.github.io/) | [Paper](https://echo-bench.github.io/) | [Data](https://huggingface.co/datasets/echo-bench/echo-bench)

This repository contains the code for the paper [**Constantly Improving Image Models Need Constantly Improving Benchmarks**](https://echo-benchmark.github.io) by [Jiaxin Ge*](https://jiaxin.ge/), [Grace Luo*](https://graceluo.net/), [Heekyung Lee](https://kyunnilee.github.io/), Nishant Malpani, [Long Lian](https://tonylian.com/), [XuDong Wang](https://people.eecs.berkeley.edu/~xdwang/), [Aleksander Holynski](https://holynski.org/), [Trevor Darrell](http://people.eecs.berkeley.edu/~trevor/), [Sewon Min](https://www.sewonmin.com/), and [David M. Chan](https://dchan.cc/).

---

## About the Dataset
ECHO is a framework for constructing benchmarks directly from social media posts, which showcase novel prompts and qualitative user judgements. As a case study, we apply ECHO to discussion of GPT-4o Image Gen on Twitter/X. Below we describe the data provided in this initial release.

We provide the academic version of our dataset, which contains the data studied in our paper, in the following HuggingFace repo: [echo-bench/echo2025](https://huggingface.co/datasets/echo-bench/echo2025).
| Split | Size | Description |
|-------|-------------|------|
| `analysis` | 29.3k | Moderate-quality data suitable for large-scale analysis.|
| `text_to_image` | 848 | High-quality data with prompt-only inputs for benchmarking.|
| `image_to_image` | 717 | High-quality data with prompt and image inputs for benchmarking.|

We also provide a version of the dataset with an MIT license in the following HuggingFace repo: [echo-bench/echo2025-mit](https://huggingface.co/datasets/echo-bench/echo2025-mit).
| Split | Size | Description |
|-------|-------------|------|
| `text_to_image` | 848 | Mirrors `text_to_image` in the academic version, but only contains the prompts extracted by our postprocessing pipeline, without other metadata.|
| `image_to_image` | 717 | Mirrors `image_to_image` in the academic version, but generates synthetic input images from dense text captions, using GPT-4o Image Gen. |

## Setup

```bash
conda env create -f environment.yaml
conda activate echo
```

## Quickstart
Load the academic version of the dataset:
```
ds = load_dataset(
    "echo-bench/echo2025",
    name="text_to_image", # ["analysis", "text_to_image", "image_to_image"]
    split="test",
)
```
Load the MIT licensed version of the dataset:
```
ds_mit = load_dataset(
    "echo-bench/echo2025-mit",
    name="text_to_image", # ["text_to_image", "image_to_image"]
    split="test",
)
```
See [`echo_bench/load_dataset.ipynb`](echo_bench/load_dataset.ipynb) for a more in-depth walkthrough for loading the dataset.

## Evaluation

All evaluation-related code is located in the `eval` folder.

### AutoEval

`autoeval` contains code for using a VLM as a judge to evaluate model outputs.  
It assigns a score to each model and then performs pairwise comparison to calculate win-rate results.

### Community-Driven Metrics

`community_driven_metrics` contains implementations of four metrics derived from community feedback.  
Each metric is located in a separate subfolder:

```bash
cd color_shift
cd face_identity
cd spatial_preservation
cd text_rendering
```

- **color_shift**: Jupyter notebook that computes color histograms and visualizes color shifts.  
- **face_identity**: Uses AuraFace to detect and calculate facial similarity between two images.  
- **spatial_preservation**: Uses Gram matrices of DINO features to measure spatial preservation.  
- **text_rendering**: Contains the prompt for using a VLM to score the text rendering quality of an image.

### Metric Assignment

The `get_metrics_assignment` folder contains the prompt used to classify which metrics apply to each sample.

- Results for the `image_to_image` assignment:  
  `eval/community_driven_metrics/image_to_image_assigment.json`
- Results for the `text_to_image` assignment:  
  `eval/community_driven_metrics/text_to_image_assigment.json`

## Data Collection Pipeline
Coming soon.

## BibTeX
```
@article{ge2025echo,
  title={Constantly Improving Image Models Need Constantly Improving Benchmarks},
  author={Jiaxin Ge, Grace Luo, Heekyung Lee, Nishant Malpani, Long Lian, XuDong Wang, Aleksander Holynski, Trevor Darrell, Sewon Min, David M. Chan},
  journal={arXiv},
  year={2025}
}
```
