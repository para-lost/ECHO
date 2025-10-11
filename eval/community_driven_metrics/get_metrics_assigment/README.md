# Metric Assignment Utilities

This folder provides utilities for determining **which community-driven metrics** each sample should be evaluated on.

## Overview

- Metric definitions are specified as YAML configuration files grouped by evaluation setting.
- `get_all_metrics.py`, classifies samples and writes the corresponding metric assignments to an output directory.

## Directory Structure

- `image_to_image_metrics/`  
  YAML configurations for:
  - **Face Identity Preservation**
  - **No Color Shift**
  - **Spatial Position Preservation**

- `text_to_image_metrics/`  
  YAML configuration for:
  - **Text Rendering Accuracy**

- `get_all_metrics.py`  
  Script that assigns metrics to samples based on the selected configuration group.

## Usage

Invoke the script with an output directory and a configuration group:

### Image-to-Image split
```bash
python get_all_metrics.py --outdir "assignments" --config "image_to_image"
```
### Text-to-Image split
```bash
python get_all_metrics.py --outdir "assignments" --config "text_to_image"
```

## Arguments

- **`--outdir`**  
  Path to the directory where metric assignment files will be written.

- **`--config`**  
  Which split to use. Supported values:
  - `image_to_image`
  - `text_to_image`
