# M4 XAI Faithfulness Benchmark: Field-Robust Trained YOLO Classifiers
## NeurIPS 2023 M4 Benchmark Protocol Implementation

| Model | Deletion AUC (↓) | Insertion AUC (↑) | Faithfulness ΔAUC (↑) | Monotonicity (↑) | Pearson $r$ (↑) | Sparsity (↑) | **M4 Composite Score** (↑) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLO11n-cls** | 0.3348 | 0.3521 | 0.0174 | 0.9057 | 0.1508 | 0.2542 | **0.3474** |
| **YOLO26n-cls** | 0.3484 | 0.3391 | -0.0092 | 0.6781 | 0.2603 | 0.1704 | **0.3161** |
| **YOLOv8n-cls** | 0.3383 | 0.3402 | 0.0020 | 0.7671 | 0.2367 | 0.1801 | **0.2604** |
