# Two-Stage Rubber Leaf Disease Project TODOs & Next Steps

This file tracks the status of data preparation, training, inference testing, web app deployment, and Explainable AI (XAI) validation.

---

## 1. Work Accomplished (Recent Progress)

### Stage 1 Detector Single-Class Transition & Literature Synthesis
- [x] **Systematic Literature Review (RRL) Synthesis**: Analyzed 18 literature papers (`D:\school\Masters\Research\RRL`). Documented academic justification for decoupling localization (single-class Stage 1 Region Proposal Network) from diagnosis (Stage 2 Classifier).
- [x] **Roboflow Dataset Rebalancing**: Created `stage-1-rubber-leaf-detection-single-class` (Version 1) with an optimal **70% Train (914 images), 20% Valid (261 images), and 10% Test (130 images)** split.
- [x] **Class 0 Removal**: Deleted ambiguous `Ignore Leaf` class in Roboflow, eliminating label noise and converting Stage 1 into a clean single-class `Rubber Leaf` detector.
- [x] **Kaggle Script Optimization**: Updated **[kaggle_stage1_training.py](file:///d:/school/Masters/Research/Training/Two%20stage%20Rubber%20Disease/Stage%201%20Rubber%20Leaf%20Detector/kaggle_stage1_training.py)**:
  - Configured 100 epochs with early stopping (`patience=15`).
  - Added dual evaluation on `val` and held-out `test` splits at operational `conf=0.25`.
  - Configured automated ONNX export (`stage1_leaf_detector.onnx`) for Hailo-8L NPU deployment.
- [x] **Validation Benchmark**: Achieved **100% Validation Recall (0 missed leaves)** on the single-class model setup.

### Stage 1 Detector Calibration, Gold Standard Test Set Evaluation & Edge Deployment Sync
- [x] **Confidence Threshold Calibration**: Calibrated detection operational threshold to **`conf=0.40`** (suppressing ~80% of outdoor background false positives while maintaining 91.36% leaf recall).
- [x] **Early Stopping Optimization**: Set training early stopping `patience=10`.
- [x] **Gold Standard Test Set Evaluation**: Standardized **[kaggle_stage1_training.py](file:///d:/school/Masters/Research/Training/Two%20stage%20Rubber%20Disease/Stage%201%20Rubber%20Leaf%20Detector/kaggle_stage1_training.py)** to evaluate on the unseen held-out **Test Set** (`plots=True, name='test_eval'`) and save `test_metrics_summary.txt` and `test_confusion_matrix.png` directly into `stage1_training_results.zip`.
- [x] **100-Epoch Training Benchmarks**:
  - **mAP@50**: **96.27%**
  - **mAP@50-95**: **84.65%**
  - **Precision ($P$)**: **90.64%**
  - **Recall ($R$)**: **91.36%**
  - **F1-Score ($F1$)**: **91.00%**
- [x] **Documentation & Methodology Updates**:
  - Recorded exact results and academic justifications in **[manuscript_notes.md](file:///d:/school/Masters/Research/Training/Two%20stage%20Rubber%20Disease/manuscript_notes.md)** (Section 5).
  - Appended new CRISP-DM Iteration log in **[CRISP_DM_Iterations.md](file:///d:/school/Masters/Research/Training/Two%20stage%20Rubber%20Disease/CRISP_DM_Iterations.md)**.
- [x] **Web Application & UI Cleanup**:
  - Updated backend `app.py` default detection threshold to `0.40`.
  - Removed legacy manual slider UI element from `index.html` and locked JS fallbacks to `0.40`.
  - Fixed local directory path `Stage 1 Rubber Leaf Detector/stage1_training_results/weights/best.pt`.
- [x] **Raspberry Pi Deployment Script (`upload_to_raspi.ps1`)**:
  - Updated path resolution (`Set-Location $PSScriptRoot`) so script executes from any terminal directory.
### Stage 2 YOLO11n-cls Field-Robust Training & Manuscript Harmonization
- [x] **Field-Robust Training**: Trained YOLO11n-cls ([kaggle_stage2_yolo11n_field_robust_training.py](file:///d:/school/Masters/Research/Training/Two%20stage%20Rubber%20Disease/Stage2_Classifier/kaggle_stage2_yolo11n_field_robust_training.py)) with Random Erasing ($40\%$), extreme lighting jitter ($V=0.70, S=0.80$), and tilt/rotation augmentations.
- [x] **Test Set Benchmarks**: Achieved **94.42% Test Accuracy**, **93.84% Macro F1**, **98.09% Healthy F1**, and **94.62% PLFD F1** on held-out test data.
- [x] **Hailo-8L Compilation**: Compiled `yolo11n_robust_best.hef` (5.49 MB) and integrated into Raspberry Pi deployment synchronization ([sync_raspi_deploy.ps1](file:///d:/school/Masters/Research/Training/Two%20stage%20Rubber%20Disease/sync_raspi_deploy.ps1)).
- [x] **Manuscript & Paper Proposal Updates**: Standardized **[manuscript_notes.md](file:///d:/school/Masters/Research/Training/Two%20stage%20Rubber%20Disease/manuscript_notes.md)**, **[paper_proposal.md](file:///d:/school/Masters/Research/Training/Two%20stage%20Rubber%20Disease/paper_proposal.md)**, and **[CRISP_DM_Iterations.md](file:///d:/school/Masters/Research/Training/Two%20stage%20Rubber%20Disease/CRISP_DM_Iterations.md)** with YOLO11n-cls metrics, architecture details, and academic justifications.

### Hardware GNSS / GPS Integration & Live Spatial Geotagging
- [x] **Hardware Installation & Verification**: Connected and verified hardware GNSS module (u-blox NEO-M8N / NEO-6M via Raspberry Pi UART /dev/ttyAMA0).
- [x] **Asynchronous Telemetry Polling**: Implemented multi-threaded NMEA parsing ($GNRMC, $GNGGA) supporting latitude, longitude, altitude, speed, and satellite count.
- [x] **Dual-Fallback Geolocation Pipeline**: Automated logging of hardware GNSS coordinates with client-side HTML5 geolocation fallback into edge inspection records and `/api/gps`.
- [x] **Persistent Field Trials Auto-Logger Engine**: Automated direct disk logging (`field_trials_master_log.csv`, `raw_frames/`, and `crops/`) on every leaf inference with endpoints (`/api/field_trials/csv` and `/api/field_trials/stats`).

---

## 2. Action Items (Next Working Days)

### 📌 Task 1: Hard Negative Mining (Background Images for Stage 1)
- [ ] Upload 50–100 background-only images (soil, tree trunks, weeds, sky with 0 bounding boxes) into Roboflow.
- [ ] Re-train Stage 1 to calculate background penalty loss, suppressing low-confidence candidate false positives.

### 📌 Task 2: Edge Hardware Benchmarking on Raspberry Pi 5 AI Kit
- [ ] Run pure Hailo-8L HEF models on Raspberry Pi 5 (`python3 app.py`).
- [ ] Benchmark live streaming throughput (FPS), end-to-end inference latency (ms), CPU/RAM load, and device thermals under field camera feeds.

### 📌 Task 3: Field Generalization & Environmental Validation
- [ ] Test distance variation (0.5m to 5.0m) to map scale degradation curves.
- [ ] Log performance under variable lighting environments (direct midday sun vs overcast shade).
