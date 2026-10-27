# Stage 2 YOLO11n-cls Field-Robust Training Results
## Master's Thesis: Two-Stage Edge AI Rubber Leaf Disease Detection

| Model | Params (M) | .pt (MB) | .onnx (MB) | Accuracy (%) | Macro P (%) | Macro R (%) | Macro F1 (%) | Wtd F1 (%) | Latency (ms) | Train (min) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **YOLO11n-cls (Field-Robust)** | 1.53 | 3.04 | 5.88 | **94.42%** | 93.55% | 94.16% | **93.84%** | 94.42% | 3.95 ms | 12.6 min |

### Per-Class Performance Breakdown

| Disease Class | Precision (%) | Recall (%) | F1-Score (%) | Test Support |
|:---|---:|---:|---:|---:|
| **Healthy** | 96.77% | 99.45% | 98.09% | 181 |
| **Other Disease** | 87.79% | 89.84% | 88.80% | 128 |
| **PLFD** | 96.09% | 93.18% | 94.62% | 264 |
