# Gracity Insects — YOLOv8 Image Classification (Notebook Project)

This repo is a **notebook-first** reference implementation to train, evaluate, and demo a YOLOv8 **classification** model on an insect image dataset stored in **OCI Object Storage**.

> This repository is **sanitized for sharing**: bucket names, namespaces, OCIDs, and local absolute paths have been replaced with placeholders like `<BUCKET_NAME>`.

## What this project delivers
- Dataset split & upload to OCI Object Storage (train / test-as-val) with reproducible structure
- Dataset QA + basic EDA (counts per class, random visual checks, image size distribution)
- YOLOv8n classification training (CPU-friendly starter config; GPU-ready with a single parameter change)
- Evaluation & metrics (accuracy + Top-K, confusion matrix, and basic latency measurements)
- Export artifacts (e.g., best weights / ONNX) and upload them back to Object Storage
- Simple Gradio demo to upload an image and get Top-1 / Top-5 predictions

## Repository structure
```
.
├── 00_environment_oci_access.ipynb
├── 01_dataset_prep_split_upload.ipynb
├── 02_dataset_qa_eda_bucket_first_fixed.ipynb
├── 03_train_yolov8n_cls.ipynb
├── 04_evaluate_metrics_latency.ipynb
├── 05_export_upload_artifacts.ipynb
├── 06_gradio_demo_upload_predict.ipynb
├── data/                # (optional) local staging area 
├── outputs/             # (optional) local analysis outputs 
└── runs/                # YOLO training outputs 
```

## Notebooks (recommended order)

### 00 — Environment & OCI access
- Validates **Resource Principal** access to Object Storage
- Lists objects, reads a small object, and writes a small test object

### 01 — Dataset preparation: split & upload
- Takes a local dataset (class folders)
- Creates a reproducible split (train + test-as-val)
- Uploads to Object Storage using a clean prefix structure

### 02 — Dataset QA & EDA (bucket-first)
- Reads from Object Storage (optionally syncs locally for speed)
- Verifies: image counts per class/split, random samples, and basic image size stats

### 03 — Train YOLOv8n classification
- Trains `yolov8n-cls` using the split dataset
- Writes outputs to `runs/`
- GPU vs CPU is controlled by a single `device` parameter

### 04 — Evaluate metrics & latency
- Loads the trained weights
- Produces classification metrics (accuracy, Top-K, confusion matrix)
- Runs basic latency benchmarking (single image and batch)

### 05 — Export & upload artifacts
- Exports the best model (e.g., ONNX)
- Uploads model artifacts + key run files back to Object Storage

### 06 — Gradio demo (upload & predict)
- Lightweight UI to upload an image and see predictions
- Useful for customer validation and quick iteration

## Configuration
Each notebook has a **Configuration** section at the top. Replace placeholders:

- `<BUCKET_NAME>`
- `<DATASET_PREFIX>`
- `<RUNS_PREFIX>`
- `<REGION>` / `<NAMESPACE>`
- `<LOCAL_PATH>`

## Notes on preprocessing
For YOLOv8 classification, heavy preprocessing is usually **not required**:
- YOLO applies resizing and augmentation internally during training.
- If images are extremely large, the best practice is to set an appropriate `imgsz` (e.g., 224/320) and let the trainer handle resizing.
- Optional: run a one-time image integrity check (corrupt files) and remove duplicates.

## Author
Cristina Varas Menadas
