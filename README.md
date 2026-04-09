# Real-Time-Acceleration-for-Medical-Image-Processing
# Breast Ultrasound AI Pipeline

This repository contains a staged ultrasound imaging pipeline that starts from raw RF data, produces beamformed images, trains classification and segmentation models, and runs patient-level predictions with visual reports.

## What Is In This Repo

- `step1_beamforming.py` converts `Dataset/OASBUD.mat` into beamformed PNG images and ROI overlays.
- `step2_enhancement.py` enhances BUSI ultrasound images with denoising, CLAHE, and bilateral filtering.
- `step_augment.py` generates a large augmented image/mask set from the beamformed images.
- `step3_cnn_train.py` trains a 3-class ResNet18 classifier on enhanced images.
- `step3b_finetune_cnn.py` fine-tunes the CNN on the augmented beamformed dataset for benign vs malignant.
- `step4_unet_train.py` trains a UNet segmentation model from paired image/mask data.
- `step4b_finetune_unet.py` fine-tunes the UNet on the augmented image/mask dataset.
- `step5_predict.py` runs the full prediction pipeline and saves visual results.
- `save_all_patients.py` runs predictions for all 100 OASBUD patients in sequence.

## Dataset Layout

The code expects the following folders:

- `Dataset/OASBUD.mat`
- `Dataset/BUSI/benign`, `Dataset/BUSI/malignant`, `Dataset/BUSI/normal`
- `Dataset/BrEaST-Lesions_USG-images_and_masks/`
- `Dataset/CIRS040GSE/`

The pipeline writes outputs into `outputs/`, including beamformed images, enhanced images, checkpoints, summaries, and prediction figures.

## Requirements

Use the project virtual environment if available. This workspace is configured to use:

- `.venv\Scripts\python.exe`

Recommended Python packages:

```bash
pip install numpy scipy matplotlib opencv-python pillow torch torchvision scikit-learn seaborn scikit-image tqdm
```

If you use VS Code, point the interpreter at `.venv\Scripts\python.exe`.

## End-to-End Workflow

Run the scripts in this order:

1. `python step1_beamforming.py`
2. `python step2_enhancement.py`
3. `python step3_cnn_train.py`
4. `python step4_unet_train.py`
5. `python step_augment.py`
6. `python step3b_finetune_cnn.py`
7. `python step4b_finetune_unet.py`
8. `python step5_predict.py --patient 0`

The augmentation step should run before the fine-tuning scripts so the CNN and UNet train on the larger augmented dataset.

## Script Details

### Step 1: Beamforming

`step1_beamforming.py` loads `Dataset/OASBUD.mat`, reconstructs ultrasound images from raw RF data, overlays lesion ROIs, and saves:

- `outputs/beamformed_images/benign/`
- `outputs/beamformed_images/malignant/`
- `outputs/beamformed_images/dataset_summary.csv`

Each patient produces plain view images, ROI overlays, and occasional comparison figures.

### Step 2: Enhancement

`step2_enhancement.py` processes BUSI images and writes enhanced images to:

- `outputs/enhanced_images/benign/`
- `outputs/enhanced_images/malignant/`
- `outputs/enhanced_images/normal/`

It also saves `outputs/enhancement_comparison.png`.

### Step 3: CNN Training

`step3_cnn_train.py` trains a ResNet18 classifier on `outputs/enhanced_images/` with three classes: benign, malignant, and normal.

Outputs:

- `outputs/model_checkpoints/best_cnn.pth`
- `outputs/results/training_curves.png`

### Step 4: UNet Training

`step4_unet_train.py` auto-detects image and mask folders under `Dataset/`, builds paired segmentation samples, and trains a binary UNet.

Outputs:

- `outputs/model_checkpoints/best_unet.pth`

### Step 5: Augmentation

`step_augment.py` expands the beamformed dataset into paired augmented images and masks.

Outputs:

- `outputs/augmented_images/benign/`
- `outputs/augmented_images/malignant/`
- `outputs/augmented_masks/benign/`
- `outputs/augmented_masks/malignant/`
- `outputs/augmented_images/augmentation_log.csv`

### Step 3b: CNN Fine-Tuning

`step3b_finetune_cnn.py` fine-tunes the ResNet18 model on the augmented beamformed images.

Outputs:

- `outputs/model_checkpoints/best_cnn_finetuned.pth`
- `outputs/results/finetune_cnn_confusion.png`
- `outputs/results/finetune_cnn_curves.png`

### Step 4b: UNet Fine-Tuning

`step4b_finetune_unet.py` fine-tunes the UNet on the augmented image/mask pairs.

Outputs:

- `outputs/model_checkpoints/best_unet_finetuned.pth`
- `outputs/results/finetune_unet_curves.png`

## Prediction Usage

`step5_predict.py` can run on OASBUD patients, a PNG image, or a synthetic `.mat` signal file.

Examples:

```bash
python step5_predict.py --patient 0
python step5_predict.py --image Dataset/BUSI/benign/benign(1).png
python step5_predict.py --signal synthetic_signals/signal_01_benign.mat
python step5_predict.py --signal all
python step5_predict.py --validate
```

If no arguments are provided, the script runs five sample OASBUD patients.

Prediction outputs are saved in `outputs/results/` as 6-panel figures that show the raw RF signal, beamformed image, enhanced image, tumor heatmap, tumor mask, and final diagnosis.

## Batch Prediction

`save_all_patients.py` loops through all 100 OASBUD patients and runs `step5_predict.py --patient N` for each one.

## Outputs

Important generated folders include:

- `outputs/beamformed_images/`
- `outputs/enhanced_images/`
- `outputs/augmented_images/`
- `outputs/augmented_masks/`
- `outputs/model_checkpoints/`
- `outputs/results/`
- `outputs/pending_appointments/`
- `outputs/pdf_reports/`

## Notes

- Run everything from the repository root: `e:\Agentic`.
- On Windows, set the interpreter to `.venv\Scripts\python.exe`.
- Some scripts expect data to exist before they run, so follow the workflow order above.
- The repository already includes trained checkpoints in `outputs/model_checkpoints/` and `kaggle_upload/`.

## Quick Start

```bash
python step1_beamforming.py
python step2_enhancement.py
python step3_cnn_train.py
python step4_unet_train.py
python step_augment.py
python step3b_finetune_cnn.py
python step4b_finetune_unet.py
python step5_predict.py --validate
```
