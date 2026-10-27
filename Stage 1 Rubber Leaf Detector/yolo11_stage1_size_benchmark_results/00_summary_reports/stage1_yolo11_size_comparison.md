# Stage 1 YOLO11 Model Size Scaling Benchmark Results
## Research Project: Master's Thesis - Edge AI Rubber Leaf Disease Detection

Comparative evaluation across **YOLO11 Nano (11n)**, **Small (11s)**, and **Medium (11m)** architectures on single-class Rubber Leaf Localization:

| Model Variant | Scale Size     | Parameters (M) | Weights (.pt) | Val mAP50 (%) | Test mAP50 (%) | Test mAP50-95 (%) | Precision (%) | Recall (%) | F1-Score (%) | GPU Latency (ms) | Train Time (min) |
|:--------------|:---------------|---------------:|:--------------|--------------:|---------------:|------------------:|--------------:|-----------:|-------------:|-----------------:|-----------------:|
| YOLO11n       | Nano (2.6M)    |           2.58 | 5.22 MB       |         97.83 |          94.70 |             85.45 |         88.34 |      91.88 |        90.08 |             7.87 |             31.1 |
| YOLO11s       | Small (9.4M)   |           9.41 | 18.29 MB      |         94.93 |          94.82 |             81.90 |         90.35 |      92.39 |        91.36 |            10.45 |             35.2 |
| YOLO11m       | Medium (20.1M) |          20.03 | 38.65 MB      |         96.16 |          95.85 |             81.83 |         87.26 |      89.34 |        88.29 |            26.08 |            110.3 |
