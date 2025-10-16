# Constantly Improving Image Models Need Constantly Improving Benchmarks

### [Project Page](https://echo-bench.github.io) | [Data](https://huggingface.co/datasets/echo-bench)

This repository contains the code accompanying the paper:

**Constantly Improving Image Models Need Constantly Improving Benchmarks**<br>
Jiaxin Ge*, Grace Luo*, Heekyung Lee, Nishant Malpani, Long Lian, XuDong Wang, Aleksander Holynski, Trevor Darrell, Sewon Min, and David M. Chan<br>
UC Berkeley

For any questions or inquiries, please contact us at [echo-bench@googlegroups.com](mailto:echo-bench@googlegroups.com).

---

## About the Dataset
ECHO stands for <u>E</u>xtracting <u>C</u>ommunity <u>H</u>atched <u>O</u>bservations. ECHO is a framework for constructing benchmarks directly from social media posts, which showcase novel prompts and qualitative user judgements. As a case study, we apply ECHO to discussion of [GPT-4o Image Gen](https://openai.com/index/introducing-4o-image-generation/) on Twitter/X. Below we describe the data provided in this initial release.

We provide the dataset in the following HuggingFace repo: [echo-bench/echo2025](https://huggingface.co/datasets/echo-bench/echo2025).
The dataset contains the following splits:
| Split | Size | Description |
|-------|-------------|------|
| `analysis` | 29.3k | Moderate-quality data suitable for large-scale analysis.|
| `text_to_image` | 848 | High-quality data with prompt-only inputs for benchmarking.|
| `image_to_image` | 710 | High-quality data with prompt and image inputs for benchmarking.|

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
See [`echo_bench/load_dataset.ipynb`](echo_bench/load_dataset.ipynb) for a more in-depth walkthrough for loading the dataset.

## Evaluation

All evaluation-related code is located in the `eval` folder.

### AutoEval

The folder `eval/autoeval` contains code for using an ensemble of VLM-as-a-judge to evaluate model outputs.
We follow the "single answer grading" setup from [MT-Bench](https://arxiv.org/abs/2306.05685).
This setup assigns a score to each output, converts these scores into pseudo pairwise comparisons, then calculates the per-model win rates.

### Community-Driven Metrics

The folder `eval/community_driven_metrics` contains implementations of four metrics derived from community feedback. These metrics correspond to each subfolder:
- `color_shift`: Computes color shift magnitude as the average difference between the color histogram of the input versus output images.
- `face_identity`: Computes face embedding similarity using [AuraFace](https://huggingface.co/fal/AuraFace-v1). 
- `spatial_preservation`: Computes structure distance using the Frobenius norm of Gram matrices derived from [DINO](https://arxiv.org/abs/2104.14294) features, implemented using the code from [Splice](https://splice-vit.github.io/).
- `text_rendering`: Computes text rendering accuracy using VLM-as-a-judge.

The `get_metrics_assignment` folder contains code for classifying which metrics apply to each sample.

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