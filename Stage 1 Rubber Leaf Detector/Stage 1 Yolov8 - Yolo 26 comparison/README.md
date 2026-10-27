# Stage 1 Single-Class Rubber Leaf Detector: Multi-Generation Benchmark

This directory contains the standardized evaluation and benchmarking results for **5 YOLO architectural generations** (YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26) evaluated on the held-out test split for single-class Rubber Leaf localization.

---

## 📂 Directory Structure

```
stage1_final_benchmark_with_RubberLeaf_matrix/
│
├── 00_summary_reports/                     # Comparative metrics in Markdown, CSV, and LaTeX
│   ├── stage1_generation_comparison.csv
│   ├── stage1_generation_comparison.md
│   └── stage1_generation_comparison_latex.tex
│
├── best_weights/                           # Consolidated .pt weights for all 5 generations
│   ├── YOLO11_best.pt
│   ├── YOLOv8_best.pt
│   ├── YOLOv9_best.pt
│   ├── YOLOv10_best.pt
│   └── YOLO26_best.pt
│
├── confusion_matrices_comparison/          # RubberLeaf confusion matrices (raw & normalized)
│   ├── YOLO11_confusion_matrix_RubberLeaf.png
│   ├── YOLO11_confusion_matrix_normalized_RubberLeaf.png
│   ├── YOLOv8_confusion_matrix_RubberLeaf.png
│   ├── YOLOv8_confusion_matrix_normalized_RubberLeaf.png
│   ├── YOLOv9_confusion_matrix_RubberLeaf.png
│   ├── YOLOv9_confusion_matrix_normalized_RubberLeaf.png
│   ├── YOLOv10_confusion_matrix_RubberLeaf.png
│   ├── YOLOv10_confusion_matrix_normalized_RubberLeaf.png
│   ├── YOLO26_confusion_matrix_RubberLeaf.png
│   └── YOLO26_confusion_matrix_normalized_RubberLeaf.png
│
├── training_curves_comparison/             # Training loss, precision, recall, and mAP curves
│   ├── YOLO11_results.png
│   ├── YOLOv8_results.png
│   ├── YOLOv9_results.png
│   ├── YOLOv10_results.png
│   └── YOLO26_results.png
│
└── per_generation_runs/                    # Full per-model outputs, training configs, and batch predictions
    ├── YOLO11/
    │   ├── weights/                        (YOLO11_best.pt, YOLO11_last.pt)
    │   ├── confusion_matrices/             (Test and validation confusion matrices)
    │   ├── logs_and_curves/                (results.csv, results.png, args.yaml)
    │   └── val_predictions/                (val_batch0/1/2 labels and predictions)
    ├── YOLOv8/
    ├── YOLOv9/
    ├── YOLOv10/
    └── YOLO26/
```

---

## 📊 Summary Benchmark Metrics

| Generation | Parameters (M) | Weights (.pt) | Val mAP50 (%) | Test mAP50 (%) | Test mAP50-95 (%) | Precision@F1max (%) | Recall@F1max (%) | F1max (%) | GPU Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11** | 2.58 | 5.22 MB | **98.22%** | **96.19%** | **92.09%** 🥇 | 92.65% | **88.83%** | **90.70%** | 8.12 ms |
| **YOLOv8** | 3.01 | 5.96 MB | 97.84% | 94.69% | 88.52% | 87.82% | **88.83%** | 88.32% | **5.79 ms** ⚡ |
| **YOLO26** | 2.38 | 5.14 MB | 97.80% | 93.82% | 90.59% | 91.93% | 86.80% | 89.29% | 10.06 ms |
| **YOLOv9** | **1.97** 🪶 | **4.43 MB** | **98.24%** 🥇 | **96.31%** 🥇 | 91.29% | **96.74%** 🥇 | 86.29% | **91.22%** 🥇 | 15.75 ms |
| **YOLOv10** | 2.27 | 5.49 MB | 98.10% | 95.15% | 89.87% | 94.16% | 82.74% | 88.08% | 7.82 ms |

---

## 🎯 Recommendations for Two-Stage Architecture & Edge AI Deployment
1. **Primary Model for Highest Localization Quality (Feeding Stage 2):** **YOLO11n** 
   - Achieves the highest **Test mAP50-95 (92.09%)**, producing the cleanest bounding boxes with minimal background clutter.
2. **Ultra-Low-Latency Edge Deployment:** **YOLOv8n**
   - Fastest inference latency (**5.79 ms**) with proven native HailoRT `.hef` compilation on the Raspberry Pi 5 + Hailo-8L NPU.
