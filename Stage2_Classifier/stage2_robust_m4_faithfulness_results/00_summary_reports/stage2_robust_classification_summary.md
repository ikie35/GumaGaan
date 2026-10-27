# Stage 2: Field-Robust Classifier Benchmark Results

## Overall Performance

| model       |   params_m |   pt_size_mb |   onnx_size_mb |   test_accuracy |   macro_precision |   macro_recall |   macro_f1 |   weighted_precision |   weighted_recall |   weighted_f1 |   latency_ms |
|:------------|-----------:|-------------:|---------------:|----------------:|------------------:|---------------:|-----------:|---------------------:|------------------:|--------------:|-------------:|
| YOLOv8n-cls |    1.44213 |      2.83173 |        5.5163  |         94.5899 |           94.3729 |        93.6894 |    93.993  |              94.5474 |           94.5899 |       94.5407 |      2.99816 |
| YOLO11n-cls |    1.53495 |      3.04346 |        5.88028 |         94.4154 |           93.5514 |        94.1577 |    93.837  |              94.4529 |           94.4154 |       94.4154 |      4.61739 |
| YOLO26n-cls |    1.53495 |      3.04334 |        5.88028 |         95.1134 |           95.1604 |        94.1445 |    94.6059 |              95.1027 |           95.1134 |       95.0727 |      4.50808 |

## Per-Class Breakdown (Precision / Recall / F1)

| model       | class         |   precision |   recall |   f1_score |   support |
|:------------|:--------------|------------:|---------:|-----------:|----------:|
| YOLOv8n-cls | Healthy       |     96.2366 |  98.895  |    97.5477 |       181 |
| YOLOv8n-cls | Other Disease |     92.5    |  86.7188 |    89.5161 |       128 |
| YOLOv8n-cls | PLFD          |     94.382  |  95.4545 |    94.9153 |       264 |
| YOLO11n-cls | Healthy       |     96.7742 |  99.4475 |    98.0926 |       181 |
| YOLO11n-cls | Other Disease |     87.7863 |  89.8438 |    88.8031 |       128 |
| YOLO11n-cls | PLFD          |     96.0938 |  93.1818 |    94.6154 |       264 |
| YOLO26n-cls | Healthy       |     97.2678 |  98.3425 |    97.8022 |       181 |
| YOLO26n-cls | Other Disease |     94.1176 |  87.5    |    90.6883 |       128 |
| YOLO26n-cls | PLFD          |     94.0959 |  96.5909 |    95.3271 |       264 |