# Two-Stage Rubber Leaf Disease Detection & Classification

This repository contains a two-stage deep learning pipeline for identifying diseases in rubber tree leaves. It is designed to be lightweight and fast, making it ideal for deployment on edge devices like the Raspberry Pi AI Kit (Hailo-8L).

## Pipeline Architecture

1. **Stage 1 (Leaf Detection)**: A YOLOv8 object detection model identifies and draws bounding boxes around all rubber leaves in a given input image or video frame. It filters out irrelevant background elements (like soil, clouds, or ignore classes).
2. **Stage 2 (Disease Classification)**: A YOLOv8-cls classification model classifies each cropped leaf image from Stage 1 into one of three categories:
   - `Healthy`
   - `PLFD` (Pestalotiopsis Leaf Fall Disease)
   - `Other Disease`

---

## Directory Structure

```
Two stage Rubber Disease/
├── Stage 1 Rubber Leaf Detector/        # Stage 1: Leaf Bounding Box Detection
│   ├── Rubber-Leaf-Disease-Detection-10/ # Roboflow downloaded detection dataset (YOLO format)
│   ├── stage1_training_results/         # Extracted training results (weights, graphs, curves)
│   ├── train_stage1.py                  # Local Python script to train Stage 1
│   ├── kaggle_training_notebook.ipynb   # Jupyter Notebook for Kaggle Stage 1 training
│   └── inference_stage1.py              # Local inference script for Stage 1 detector only
│
├── Stage2_Classifier/                   # Stage 2: Cropped Leaf Disease Classification
│   ├── stage2_training_results/         # Extracted training results (weights, graphs, curves)
│   ├── train_stage2.py                  # Local Python script to train Stage 2
│   ├── kaggle_stage2_training.py        # Python cell script for Kaggle Stage 2 training
│   └── kaggle_stage2_notebook.ipynb     # Jupyter Notebook for Kaggle Stage 2 training
│
├── dataset_tools/                       # Utilities and dataset preparation scripts
│   ├── check_roboflow_filenames.py      # Inspect downloaded filenames from Roboflow
│   ├── crop_dataset.py                  # Model-based cropping utility (using Stage 1 weights)
│   └── crop_dataset_gt.py               # Ground-truth-based cropping utility (using annotation coordinates)
│
├── inference_two_stage.py               # Unified end-to-end inference (Detector -> Crop -> Classifier)
├── README.md                            # Project documentation (this file)
└── TODO.md                              # Consolidated task list & next steps
```

---

## Installation & Setup

Ensure you have Python 3.8+ installed. Install the necessary libraries:

```bash
pip install ultralytics roboflow opencv-python numpy
```

---

## Usage Guide

### 1. Training Stage 1 (Leaf Detection)
You can train the leaf detector either locally or on Kaggle:
- **Local**: Run `python "Stage 1 Rubber Leaf Detector/train_stage1.py"`.
- **Kaggle**: Upload and run the cells in `Stage 1 Rubber Leaf Detector/kaggle_training_notebook.ipynb` or copy-paste cells from `Stage 1 Rubber Leaf Detector/kaggle_stage1_training.py`.
- **Note**: Ensure that after training, the results folder `stage1_training_results` is downloaded and placed in the `Stage 1 Rubber Leaf Detector` directory.

### 2. Preparing Dataset for Stage 2
To train the classifier, you need a dataset consisting of cropped leaf images grouped by category:
- **Ground Truth Bounding Box Cropping**: Run `python dataset_tools/crop_dataset_gt.py`. This reads annotation coordinates from Roboflow dataset and crops original images into `D:\school\Masters\Research\ChatGPT\Dataset_Cropped_GT`.
- **Model Bounding Box Cropping**: Run `python dataset_tools/crop_dataset.py`. This uses the trained Stage 1 detector model to predict and crop leaves from any unannotated raw image folders.

### 3. Training Stage 2 (Disease Classification)
- **Local**: Run `python Stage2_Classifier/train_stage2.py`.
- **Kaggle**: Upload and run `Stage2_Classifier/kaggle_stage2_notebook.ipynb` or copy-paste cells from `Stage2_Classifier/kaggle_stage2_training.py`.
- **Note**: Ensure that after training, the results folder `stage2_training_results` is downloaded and placed in the `Stage2_Classifier` directory.

### 4. Running End-to-End Inference
To run the full two-stage pipeline on an image, a video, or an entire directory of images:

```bash
# Run on an image file
python inference_two_stage.py --source path/to/image.jpg

# Run on a video file
python inference_two_stage.py --source path/to/video.mp4

# Run on a directory of images
python inference_two_stage.py --source path/to/image_folder/

# Use GPU (if CUDA is configured)
python inference_two_stage.py --source path/to/image.jpg --device 0
```

Annotated output files with bounding boxes color-coded by disease category (Green = Healthy, Red = PLFD, Orange = Other) will be saved to `runs/detect/two_stage/`.

---

## Deployment (ONNX & Hailo)
Both training pipelines automatically export the trained models to `.onnx` format:
- ONNX models are saved in the `weights` folders alongside `best.pt`.
- These ONNX files can be compiled using the Hailo Dataflow Compiler (DFC) to `.hef` format for hardware-accelerated real-time inference on the Raspberry Pi AI Kit.
