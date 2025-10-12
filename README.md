# Constantly Improving Image Models Need Constantly Improving Benchmarks

### [Project Page](https://echo-bench.github.io/) | [Paper](https://echo-bench.github.io/) | [Data](https://huggingface.co/datasets/echo-bench/echo-bench)

This repository contains the code for the paper [**Constantly Improving Image Models Need Constantly Improving Benchmarks**](https://echo-benchmark.github.io) by [Jiaxin Ge*](https://jiaxin.ge/), [Grace Luo*](https://graceluo.net/), [Heekyung Lee](https://kyunnilee.github.io/), Nishant Malpani, [Long Lian](https://tonylian.com/), [XuDong Wang](https://people.eecs.berkeley.edu/~xdwang/), [Aleksander Holynski](https://holynski.org/), [Trevor Darrell](http://people.eecs.berkeley.edu/~trevor/), [Sewon Min](https://www.sewonmin.com/), and [David M. Chan](https://dchan.cc/).

---

## Setup

```bash
conda env create -f environment.yaml
conda activate echo
```

## Download the Dataset
Inside the echo_bench folder:

1. Download images for the `image_to_image` split:
```
cd echo_bench
python download_script.py
```

2. Load datasets with `load_dataset.ipynb`, which contains code for loading:

- `image_to_image`
- `text_to_image`
- `image_to_image_synthetic`
- `analysis`

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


## Scraping Pipeline and Postprocess Pipeline
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
