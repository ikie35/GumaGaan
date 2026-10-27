# Stage 2 Rubber Leaf Disease Classifier: Cross-Generation Benchmark
## Master's Thesis Research: Two-Stage Edge AI Rubber Leaf Disease Detection

Comparative evaluation across **YOLOv8-cls**, **YOLO11-cls**, and **YOLO26-cls** on the
independent held-out test split (3 disease categories: *Healthy*, *Other Disease*, *PLFD*):

### 1. Overall Architectural Performance Summary

| Generation | Params (M) | .pt (MB) | .onnx (MB) | Accuracy (%) | Macro P (%) | Macro R (%) | Macro F1 (%) | Wtd F1 (%) | Latency (ms) | Train (min) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **yolov8n-cls** | 1.44 | 2.83 | 5.52 | **94.59%** | 95.32% | 93.09% | **93.99%** | 94.50% | 2.51 ms | 5.3 min |
| **yolo11n-cls** | 1.53 | 3.04 | 5.88 | **94.24%** | 93.70% | 93.59% | **93.65%** | 94.24% | 4.13 ms | 12.3 min |
| **yolo26n-cls** | 1.53 | 3.04 | 5.88 | **94.24%** | 93.76% | 93.65% | **93.70%** | 94.23% | 3.90 ms | 11.3 min |

### 2. Per-Class Diagnosis Performance (Precision, Recall, F1-Score)

| Generation | Disease Category | Precision (%) | Recall (%) | F1-Score (%) | Test Support |
|:---|:---|---:|---:|---:|---:|
| yolov8n-cls | Healthy | 94.68% | 98.34% | 96.48% | 181 |
| yolov8n-cls | Other Disease | 98.17% | 83.59% | 90.30% | 128 |
| yolov8n-cls | PLFD | 93.12% | 97.35% | 95.19% | 264 |
| yolo11n-cls | Healthy | 97.79% | 97.79% | 97.79% | 181 |
| yolo11n-cls | Other Disease | 88.98% | 88.28% | 88.63% | 128 |
| yolo11n-cls | PLFD | 94.34% | 94.70% | 94.52% | 264 |
| yolo26n-cls | Healthy | 97.27% | 98.34% | 97.80% | 181 |
| yolo26n-cls | Other Disease | 89.68% | 88.28% | 88.98% | 128 |
| yolo26n-cls | PLFD | 94.32% | 94.32% | 94.32% | 264 |
