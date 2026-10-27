# Stage 1 Rubber Leaf Detector — Standard Benchmark Results

## Evaluation Protocol
- Dataset: Held-out **test split** (never seen during training or hyperparameter tuning)
- mAP@50 and mAP@50–95: per-class COCO standard via Ultralytics `model.val()`
- All metrics reported for **Class 0: Rubber Leaf** only (target class)
- Precision, Recall, F1: at the **F1-max confidence threshold** (best operating point)
- Class 1 (Ignore Leaf) excluded — it is an intentionally noisy background anchor class
- `GPU_T4_Latency_ms` = Kaggle T4 GPU inference time — NOT edge-device latency

## Results

| Model Variant   | Size Label     |   Parameters (M) |   Val mAP50_RubberLeaf (%) |   Test mAP50_RubberLeaf (%) |   Test mAP50-95_RubberLeaf (%) |   Precision_at_F1max (%) |   Recall_at_F1max (%) |   F1max (%) |   GPU_T4_Latency_ms | ONNX File                                |
|:----------------|:---------------|-----------------:|---------------------------:|----------------------------:|-------------------------------:|-------------------------:|----------------------:|------------:|--------------------:|:-----------------------------------------|
| YOLOv8n         | Nano (3.2M)    |             3.01 |                      97.9  |                       95.31 |                          87.11 |                    92.4  |                 86.45 |       89.33 |                4.47 | stage1_single_class_yolov8n_dynamic.onnx |
| YOLOv8s         | Small (11.2M)  |            11.13 |                      95.88 |                       94.77 |                          84.54 |                    92.9  |                 86.34 |       89.5  |                9.8  | stage1_single_class_yolov8s_dynamic.onnx |
| YOLOv8m         | Medium (25.9M) |            25.84 |                      96.54 |                       95.23 |                          87.43 |                    95.27 |                 84.26 |       89.43 |               24.67 | stage1_single_class_yolov8m_dynamic.onnx |
