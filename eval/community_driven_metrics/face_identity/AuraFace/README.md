# Evaluate Face Similarity with AuraFace

Scripts for using the AuraFace model is under `eval/AuraFace`. The script supports multiple input/output images with maximum similarity scoring.

## Setup - Install dependencies and model

### 1. Install Dependencies for AuraFace

```bash
pip install -r requirements.txt
```

### 2. How to download the model (AuraFace)

```bash
python download_model.py
```

This will download the AuraFace-v1 model.


## How to use the model

### Get all the face identity results

```bash
./run_face_identity_all.sh
```

