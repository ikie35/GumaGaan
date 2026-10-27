# Stage 2 YOLO11-cls Scale Comparison Benchmark
## Masters Thesis Research: Two-Stage Edge AI Rubber Leaf Disease Detection

| Scale | Model | Params (M) | .pt (MB) | .onnx (MB) | Accuracy (%) | Macro P (%) | Macro R (%) | Macro F1 (%) | Wtd F1 (%) | Latency (ms) | Train (min) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Nano** | yolo11n-cls | 1.53 | 3.04 | 5.88 | **94.24%** | 93.70% | 93.59% | **93.65%** | 94.24% | 3.53 ms | 12.3 min |
| **Small** | yolo11s-cls | 5.45 | 10.52 | 20.79 | **93.72%** | 93.57% | 92.90% | **93.22%** | 93.71% | 3.92 ms | 6.5 min |
| **Medium** | yolo11m-cls | 10.36 | 19.92 | 39.52 | **94.42%** | 94.45% | 93.52% | **93.96%** | 94.39% | 4.91 ms | 14.7 min |

| Scale | Model | Disease Class | Precision (%) | Recall (%) | F1-Score (%) | Test Support |
|:---|:---|:---|---:|---:|---:|---:|
| Nano | yolo11n-cls | Healthy | 97.79% | 97.79% | 97.79% | 181 |
| Nano | yolo11n-cls | Other Disease | 88.98% | 88.28% | 88.63% | 128 |
| Nano | yolo11n-cls | PLFD | 94.34% | 94.70% | 94.52% | 264 |
| Small | yolo11s-cls | Healthy | 97.75% | 96.13% | 96.94% | 181 |
| Small | yolo11s-cls | Other Disease | 90.32% | 87.50% | 88.89% | 128 |
| Small | yolo11s-cls | PLFD | 92.62% | 95.08% | 93.83% | 264 |
| Medium | yolo11m-cls | Healthy | 97.78% | 97.24% | 97.51% | 181 |
| Medium | yolo11m-cls | Other Disease | 92.56% | 87.50% | 89.96% | 128 |
| Medium | yolo11m-cls | PLFD | 93.01% | 95.83% | 94.40% | 264 |
