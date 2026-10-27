import os
import sys

# Force Ultralytics into offline mode to prevent telemetry/checks
os.environ["YOLO_OFFLINE"] = "True"

import cv2
import math
import threading
import time
import atexit
import uuid
import csv
from datetime import datetime
from collections import deque
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO, settings

# Disable online settings
try:
    settings.update({'sync': False})
except Exception as e:
    print(f"Warning: Could not update Ultralytics settings: {e}")

app = Flask(__name__)

camera_lock = threading.Lock()
global_camera = None

# Camera index — 0 is the only Pi Camera Module 3 Wide (Arducam removed)
PI_CAMERA_INDEX = int(os.environ.get("CAM_INDEX", "0"))

class PiCameraWrapper:
    """Seamless hybrid camera wrapper: Picamera2 (native Bookworm) -> OpenCV VideoCapture."""
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.backend = None
        self.picam2 = None
        self.cap = None
        self._init_camera()

    def _init_camera(self):
        # 1. Try native Picamera2 (Raspberry Pi OS Bookworm official stack)
        try:
            from picamera2 import Picamera2
            p2 = Picamera2()
            config = p2.create_preview_configuration(
                main={"size": (1280, 720), "format": "BGR888"}
            )
            p2.configure(config)
            try:
                # Enable continuous autofocus for Pi Camera Module 3
                p2.set_controls({"AfMode": 2})
            except Exception:
                pass
            p2.start()
            time.sleep(0.5)
            self.picam2 = p2
            self.backend = "picam2"
            print("[Camera] Pi Camera Module 3 opened via native Picamera2 (with continuous autofocus)! ✅")
            return
        except Exception as e:
            print(f"[Camera] Picamera2 unavailable or failed ({e}), falling back to OpenCV VideoCapture...")

        # 2. Try OpenCV VideoCapture (V4L2 or libcamerify)
        try:
            print(f"[Camera] Opening camera via OpenCV VideoCapture (index={self.camera_id})...")
            cap = cv2.VideoCapture(self.camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                self.cap = cap
                self.backend = "opencv"
                print(f"[Camera] Pi Camera opened successfully via OpenCV (index={self.camera_id})! ✅")
            else:
                print("[Camera] ERROR: Could not open camera. If on Pi 5, launch with: libcamerify python3 app.py")
        except Exception as e:
            print(f"[Camera] OpenCV VideoCapture error: {e}")

    def is_opened(self):
        if self.backend == "picam2" and self.picam2 is not None:
            return True
        if self.backend == "opencv" and self.cap is not None:
            return self.cap.isOpened()
        return False

    def isOpened(self):
        return self.is_opened()

    def read(self):
        if self.backend == "picam2" and self.picam2 is not None:
            try:
                frame = self.picam2.capture_array()
                return True, frame
            except Exception as e:
                print(f"[Camera] Picamera2 frame grab error: {e}")
                return False, None
        elif self.backend == "opencv" and self.cap is not None:
            return self.cap.read()
        return False, None

    def release(self):
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.backend = None

def get_camera(camera_id=None):
    global global_camera
    if camera_id is None:
        camera_id = PI_CAMERA_INDEX
    if global_camera is None or not global_camera.isOpened():
        global_camera = PiCameraWrapper(camera_id)
    return global_camera if global_camera.isOpened() else None

def read_camera_frame():
    global global_camera
    cam = get_camera()
    if cam is not None and cam.isOpened():
        return cam.read()
    return False, None

def release_camera():
    global global_camera
    if global_camera is not None:
        print("[Camera] Releasing camera...")
        try:
            global_camera.release()
        except Exception as e:
            print(f"[Camera] Error releasing: {e}")
        global_camera = None

atexit.register(release_camera)
app.config['UPLOAD_FOLDER'] = Path(__file__).resolve().parent / "uploads"
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# ─────────────────────────────────────────────────────────────
#  Field Trials Persistent Auto-Logger Engine
# ─────────────────────────────────────────────────────────────
FIELD_LOGS_DIR = Path(__file__).resolve().parent / "logs" / "field_trials"
FIELD_LOGS_RAW = FIELD_LOGS_DIR / "raw_frames"
FIELD_LOGS_CROPS = FIELD_LOGS_DIR / "crops"
FIELD_MASTER_CSV = FIELD_LOGS_DIR / "field_trials_master_log.csv"

FIELD_LOGS_RAW.mkdir(parents=True, exist_ok=True)
FIELD_LOGS_CROPS.mkdir(parents=True, exist_ok=True)

CSV_LOCK = threading.Lock()
CSV_FIELDNAMES = [
    "Timestamp",
    "Session_ID",
    "Sample_Tag_ID",
    "Tree_Clone",
    "Canopy_Level",
    "Ground_Truth_Class",
    "Disease_Severity",
    "Lighting_Condition",
    "Lux_Level_lx",
    "Distance_meters",
    "Angle_degrees",
    "Wind_Condition",
    "Camera_Sensor",
    "GPS_Latitude",
    "GPS_Longitude",
    "GPS_Altitude_m",
    "GPS_Satellites",
    "GPS_Fix_Status",
    "GPS_Source",
    "Stage1_Detected",
    "Stage1_Confidence",
    "Stage1_BBox_Coords",
    "Stage2_Pred_Class",
    "Stage2_Confidence",
    "Stage1_Latency_ms",
    "Stage2_Latency_ms",
    "Total_Latency_ms",
    "Raw_Image_Path",
    "Crop_Image_Path"
]

def init_field_master_csv():
    with CSV_LOCK:
        if not FIELD_MASTER_CSV.exists() or FIELD_MASTER_CSV.stat().st_size == 0:
            try:
                with open(FIELD_MASTER_CSV, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                    writer.writeheader()
            except Exception as e:
                print(f"[FieldLogger] Init CSV error: {e}")

init_field_master_csv()

def append_field_trial_log(row_data):
    with CSV_LOCK:
        try:
            file_exists = FIELD_MASTER_CSV.exists() and FIELD_MASTER_CSV.stat().st_size > 0
            with open(FIELD_MASTER_CSV, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_data)
        except Exception as e:
            print(f"[FieldLogger] Error writing CSV log: {e}")

detector = None
classifier = None
hailo_detector = None
hailo_classifier = None

# ─────────────────────────────────────────────────────────────
#  Hardware GNSS / GPS Manager (u-blox NEO-M8N via /dev/ttyAMA0)
# ─────────────────────────────────────────────────────────────
latest_gps_data = {
    "status": "searching",   # "locked", "searching", "unavailable"
    "latitude": None,
    "longitude": None,
    "altitude_m": None,
    "satellites": 0,
    "speed_kmh": None,
    "source": "Hardware NEO-M8N"
}
gps_lock = threading.Lock()

def parse_nmea_coord(raw_coord, direction):
    """Converts NMEA ddmm.mmmm format to decimal degrees."""
    if not raw_coord or not direction:
        return None
    try:
        if direction in ['N', 'S']:
            degrees = float(raw_coord[:2])
            minutes = float(raw_coord[2:])
        else:
            degrees = float(raw_coord[:3])
            minutes = float(raw_coord[3:])
        decimal = degrees + (minutes / 60.0)
        if direction in ['S', 'W']:
            decimal = -decimal
        return round(decimal, 6)
    except Exception:
        return None

def gps_background_worker():
    global latest_gps_data
    candidate_ports = [
        os.environ.get("GPS_PORT", "/dev/ttyAMA0"),
        "/dev/serial0",
        "/dev/ttyUSB0",
        "/dev/ttyACM0"
    ]
    
    ser = None
    try:
        import serial
    except ImportError:
        print("[GPS] Warning: 'pyserial' not installed. Hardware GPS polling disabled.")
        with gps_lock:
            latest_gps_data["status"] = "unavailable"
        return

    while True:
        if ser is None or not ser.is_open:
            for port in candidate_ports:
                try:
                    if Path(port).exists() or sys.platform.startswith("win"):
                        ser = serial.Serial(port, 9600, timeout=1.5)
                        print(f"[GPS] Connected to GNSS module on {port} at 9600 baud.")
                        break
                except Exception:
                    ser = None
            if ser is None:
                with gps_lock:
                    latest_gps_data["status"] = "unavailable"
                time.sleep(3.0)
                continue

        try:
            raw_line = ser.readline()
            if not raw_line:
                continue
            line = raw_line.decode('ascii', errors='replace').strip()
            if not (line.startswith('$') and '*' in line):
                continue
            
            # Parse $GNRMC / $GPRMC
            if line.startswith(('$GNRMC', '$GPRMC')):
                parts = line.split(',')
                if len(parts) >= 7:
                    status_flag = parts[2]  # 'A' = Active, 'V' = Void / Searching
                    with gps_lock:
                        if status_flag == 'A':
                            lat = parse_nmea_coord(parts[3], parts[4])
                            lon = parse_nmea_coord(parts[5], parts[6])
                            if lat is not None and lon is not None:
                                latest_gps_data["latitude"] = lat
                                latest_gps_data["longitude"] = lon
                                latest_gps_data["status"] = "locked"
                                latest_gps_data["source"] = "Hardware NEO-M8N"
                            if len(parts) >= 8 and parts[7]:
                                try:
                                    latest_gps_data["speed_kmh"] = round(float(parts[7]) * 1.852, 1)
                                except ValueError:
                                    pass
                        else:
                            latest_gps_data["status"] = "searching"
                            latest_gps_data["source"] = "Hardware NEO-M8N"
                            
            # Parse $GNGGA / $GPGGA
            elif line.startswith(('$GNGGA', '$GPGGA')):
                parts = line.split(',')
                if len(parts) >= 10:
                    with gps_lock:
                        try:
                            sats = int(parts[7]) if parts[7] else 0
                            latest_gps_data["satellites"] = sats
                        except ValueError:
                            pass
                        try:
                            alt = float(parts[9]) if parts[9] else None
                            latest_gps_data["altitude_m"] = alt
                        except ValueError:
                            pass
                        if parts[6] and parts[6] != '0':
                            lat = parse_nmea_coord(parts[2], parts[3])
                            lon = parse_nmea_coord(parts[4], parts[5])
                            if lat is not None and lon is not None:
                                latest_gps_data["latitude"] = lat
                                latest_gps_data["longitude"] = lon
                                latest_gps_data["status"] = "locked"
                                latest_gps_data["source"] = "Hardware NEO-M8N"

        except Exception as e:
            try:
                if ser:
                    ser.close()
            except Exception:
                pass
            ser = None
            time.sleep(2.0)

gps_thread = threading.Thread(target=gps_background_worker, daemon=True)
gps_thread.start()


# ─────────────────────────────────────────────────────────────
#  Hailo-8L NPU Inference Engine (uses .hef files via HailoRT)
# ─────────────────────────────────────────────────────────────
# Singleton shared VDevice for Hailo-8L (one physical device, multiple models via scheduler)
_hailo_vdevice = None

def get_hailo_vdevice():
    global _hailo_vdevice
    if _hailo_vdevice is None:
        from hailo_platform import VDevice, HailoSchedulingAlgorithm
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        _hailo_vdevice = VDevice(params)
        print("[Hailo] Shared VDevice created with ROUND_ROBIN scheduler")
    return _hailo_vdevice

class HailoInference:
    def __init__(self, hef_path):
        from hailo_platform import (HEF, HailoStreamInterface,
                                     InferVStreams, ConfigureParams,
                                     InputVStreamParams, OutputVStreamParams,
                                     FormatType)
        self.hef = HEF(str(hef_path))
        self.target = get_hailo_vdevice()  # Shared device
        configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_groups = self.target.configure(self.hef, configure_params)
        self.network_group = self.network_groups[0]
        input_vstreams_params = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8)
        output_vstreams_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32)
        info = self.hef.get_input_vstream_infos()[0]
        self.input_name = info.name
        self.input_h = info.shape[0]
        self.input_w = info.shape[1]

        # Open InferVStreams ONCE and keep it persistent (much faster than re-opening per frame)
        self._pipeline = InferVStreams(self.network_group,
                                       input_vstreams_params,
                                       output_vstreams_params)
        self._pipeline.__enter__()
        print(f"[Hailo] Loaded {hef_path.name} — input {self.input_h}x{self.input_w} | Pipeline ready")

    def infer(self, bgr_image):
        """Run inference on Hailo-8L NPU using persistent pipeline with high-resolution interpolation and RGB conversion."""
        import numpy as np
        h, w = bgr_image.shape[:2]
        # Use INTER_AREA for downscaling and INTER_CUBIC for upscaling to preserve sharp lesion borders
        interp = cv2.INTER_AREA if (w > self.input_w or h > self.input_h) else cv2.INTER_CUBIC
        resized = cv2.resize(bgr_image, (self.input_w, self.input_h), interpolation=interp)
        
        # Convert BGR (OpenCV) to RGB (YOLO/Hailo training format)
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Hailo NPU expects (1, H, W, 3) uint8 image array
        batch = np.expand_dims(rgb_image, axis=0)
        return self._pipeline.infer({self.input_name: batch})

    def close(self):
        try:
            self._pipeline.__exit__(None, None, None)
        except Exception:
            pass

def release_hailo_vdevice():
    global _hailo_vdevice
    if _hailo_vdevice is not None:
        try:
            _hailo_vdevice.release()
        except Exception:
            pass
        _hailo_vdevice = None

atexit.register(release_hailo_vdevice)

def hailo_postprocess_detector(raw_outputs, orig_w, orig_h, conf_thresh=0.15):
    """Parse Hailo-8L YOLO detector NMS output into (x1, y1, x2, y2, conf) list.
    Hailo NMS outputs proposals formatted as [ymin, xmin, ymax, xmax, confidence] normalized [0, 1].
    """
    import numpy as np
    boxes = []
    
    for name, tensor in raw_outputs.items():
        arr = np.array(tensor)
        if arr.size == 0:
            continue
            
        if arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim == 1:
            if arr.shape[0] >= 5:
                arr = arr.reshape(1, -1)
            else:
                continue

        for det in arr:
            if len(det) < 5:
                continue
                
            conf = float(det[4])
            if conf < conf_thresh or conf <= 0.01:
                continue

            ymin, xmin, ymax, xmax = float(det[0]), float(det[1]), float(det[2]), float(det[3])
            
            # Check coordinate format
            if max(ymin, xmin, ymax, xmax) <= 1.0:
                # Normalized coordinates (0.0 - 1.0)
                if ymax > ymin and xmax > xmin:
                    # Standard Hailo NMS format: [ymin, xmin, ymax, xmax]
                    x1 = int(xmin * orig_w)
                    y1 = int(ymin * orig_h)
                    x2 = int(xmax * orig_w)
                    y2 = int(ymax * orig_h)
                else:
                    # Center format: (cx, cy, w, h)
                    cx, cy, w_box, h_box = ymin, xmin, ymax, xmax
                    x1 = int((cx - w_box / 2.0) * orig_w)
                    y1 = int((cy - h_box / 2.0) * orig_h)
                    x2 = int((cx + w_box / 2.0) * orig_w)
                    y2 = int((cy + h_box / 2.0) * orig_h)
            else:
                # Pixel coordinates (0 - 640)
                scale_x = orig_w / 640.0
                scale_y = orig_h / 640.0
                if ymax > ymin and xmax > xmin:
                    x1 = int(xmin * scale_x)
                    y1 = int(ymin * scale_y)
                    x2 = int(xmax * scale_x)
                    y2 = int(ymax * scale_y)
                else:
                    x1 = int((ymin - ymax / 2.0) * scale_x)
                    y1 = int((xmin - xmax / 2.0) * scale_y)
                    x2 = int((ymin + ymax / 2.0) * scale_x)
                    y2 = int((xmin + xmax / 2.0) * scale_y)

            # Ensure valid bounds
            x1, y1 = max(0, min(x1, orig_w - 1)), max(0, min(y1, orig_h - 1))
            x2, y2 = max(0, min(x2, orig_w)), max(0, min(y2, orig_h))

            if x2 > x1 + 5 and y2 > y1 + 5:
                boxes.append((x1, y1, x2, y2, conf))
                
    if len(boxes) > 0:
        print(f"[Stage 1 Detector] Found {len(boxes)} leaf candidate(s)")
        
    return boxes

def hailo_postprocess_classifier(raw_outputs):
    """Parse Hailo classifier output → (class_index, confidence) using Softmax scaling and disease sensitivity calibration."""
    import numpy as np
    for name, tensor in raw_outputs.items():
        arr = np.squeeze(tensor).astype(np.float32)
        if arr.ndim == 1 and len(arr) >= 3:
            # Apply Softmax normalization
            exp_arr = np.exp(arr - np.max(arr))
            probs = exp_arr / np.sum(exp_arr)
            
            # Disease Sensitivity Calibration: If PLFD (2) or Other Disease (1) has >=20% probability, flag disease!
            plfd_prob = probs[2]
            other_prob = probs[1]
            healthy_prob = probs[0]
            
# ─────────────────────────────────────────────────────────────
#  Leaf Tracking & Temporal Stabilization Engine
# ─────────────────────────────────────────────────────────────
leaf_id_counter = 0
tracked_leaves = {}
CLASS_NAMES = {0: "Healthy", 1: "Other Disease", 2: "PLFD"}

class LeafTrack:
    """Maintains rock-steady temporal continuity, spatial EMA smoothing, and probabilistic stability."""
    def __init__(self, x1, y1, x2, y2, raw_label, conf, probs):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.cx = (float(x1) + float(x2)) / 2.0
        self.cy = (float(y1) + float(y2)) / 2.0
        self.conf = float(conf)
        self.probs = dict(probs) if probs else {"Healthy": 1.0, "Other Disease": 0.0, "PLFD": 0.0}
        self.label = raw_label
        self.lost_frames = 0
        self.seen_frames = 1

    def update(self, x1, y1, x2, y2, raw_label, conf, probs, alpha_pos=0.35, alpha_prob=0.20):
        # 1. Spatial Coordinate EMA smoothing (eliminates box jumping/jitter)
        self.x1 = alpha_pos * float(x1) + (1.0 - alpha_pos) * self.x1
        self.y1 = alpha_pos * float(y1) + (1.0 - alpha_pos) * self.y1
        self.x2 = alpha_pos * float(x2) + (1.0 - alpha_pos) * self.x2
        self.y2 = alpha_pos * float(y2) + (1.0 - alpha_pos) * self.y2
        self.cx = (self.x1 + self.x2) / 2.0
        self.cy = (self.y1 + self.y2) / 2.0
        
        # 2. Confidence score EMA smoothing
        self.conf = alpha_pos * float(conf) + (1.0 - alpha_pos) * self.conf
        
        # 3. Class probability distribution EMA smoothing
        for k in self.probs:
            self.probs[k] = alpha_prob * float(probs.get(k, 0.0)) + (1.0 - alpha_prob) * self.probs[k]
            
        # 4. Probabilistic Voting & True Argmax (No bias, stable classification)
        best_k = max(self.probs, key=self.probs.get)
        self.label = best_k
        self.lost_frames = 0
        self.seen_frames += 1

def letterbox_crop(crop):
    """Square-pad crop with letterboxing to prevent aspect ratio distortion before 224x224 classification."""
    import numpy as np
    h, w = crop.shape[:2]
    if h == w:
        return crop
    max_dim = max(h, w)
    padded = np.zeros((max_dim, max_dim, 3), dtype=crop.dtype)
    dx = (max_dim - w) // 2
    dy = (max_dim - h) // 2
    padded[dy:dy+h, dx:dx+w] = crop
    return padded

def load_models():
    global detector, classifier, hailo_detector, hailo_classifier
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"

    det_hef  = models_dir / "stage1_detector.hef"
    det_onnx = models_dir / "stage1_detector.onnx"
    det_pt   = models_dir / "stage1_detector.pt"
    if not any([det_hef.exists(), det_onnx.exists(), det_pt.exists()]):
        # Primary: YOLOv8n single-class leaf detector HEF
        det_hef_v8 = script_dir / "Stage 1 Rubber Leaf Detector" / "HEF models" / "stage1_single_class_yolov8n_best_hailo_model" / "stage1_single_class_yolov8n_best.hef"
        det_hef_v11 = script_dir / "Stage 1 Rubber Leaf Detector" / "HEF models" / "YOLO11n_best_hailo_model" / "YOLO11n_best.hef"
        det_hef  = det_hef_v8 if det_hef_v8.exists() else det_hef_v11
        det_onnx = script_dir / "Stage 1 Rubber Leaf Detector" / "stage1_training_results" / "weights" / "stage1_best.onnx"
        det_pt   = script_dir / "Stage 1 Rubber Leaf Detector" / "stage1_training_results" / "weights" / "stage1_best.pt"

    cls_hef  = models_dir / "stage2_classifier.hef"
    cls_onnx = models_dir / "stage2_classifier.onnx"
    cls_pt   = models_dir / "stage2_classifier.pt"
    if not any([cls_hef.exists(), cls_onnx.exists(), cls_pt.exists()]):
        # Primary: Field-Robust YOLO11n-cls HEF (yolo11n_robust_best.hef)
        cls_hef_v11_robust = script_dir / "Stage2_Classifier" / "yolo11n-clsHEF" / "yolo11n-cls_best_hailo_model" / "yolo11n_robust_best.hef"
        cls_hef_v11 = script_dir / "Stage2_Classifier" / "yolo11n-clsHEF" / "yolo11n-cls_best_hailo_model" / "yolo11n-cls_best.hef"
        cls_hef_v8 = script_dir / "Stage2_Classifier" / "stage2best" / "stage2_best_hailo_model" / "stage2_best.hef"
        cls_hef_v8m = script_dir / "Stage2_Classifier" / "yolov8m-clsHEF" / "yolov8m-cls_best_hailo_model" / "yolov8m-cls_best.hef"
        cls_hef  = cls_hef_v11_robust if cls_hef_v11_robust.exists() else (cls_hef_v11 if cls_hef_v11.exists() else (cls_hef_v8 if cls_hef_v8.exists() else cls_hef_v8m))
        cls_onnx = script_dir / "Stage2_Classifier" / "Training Result on v4 dataset" / "weights" / "best.onnx"
        cls_pt   = script_dir / "Stage2_Classifier" / "Training Result on v4 dataset" / "weights" / "best.pt"

    # ── Try Hailo-8L NPU (.hef) first ────────────────────────
    if det_hef.exists():
        try:
            print(f"[Model] Loading Stage 1 Detector on Hailo-8L NPU: {det_hef}")
            hailo_detector = HailoInference(det_hef)
            print("[Model] Stage 1 Detector → Hailo-8L NPU ✅")
        except Exception as e:
            print(f"[Model] Hailo Stage 1 load failed ({e}), falling back to ONNX/PT")
            hailo_detector = None

    if cls_hef.exists():
        try:
            print(f"[Model] Loading Stage 2 Classifier on Hailo-8L NPU: {cls_hef}")
            hailo_classifier = HailoInference(cls_hef)
            print("[Model] Stage 2 Classifier → Hailo-8L NPU ✅")
        except Exception as e:
            print(f"[Model] Hailo Stage 2 load failed ({e}), falling back to ONNX/PT")
            hailo_classifier = None

    # ── Fall back to ONNX / PT via Ultralytics ───────────────
    if hailo_detector is None:
        det_path = det_onnx if det_onnx.exists() else det_pt
        if det_path.exists():
            print(f"[Model] Loading Stage 1 Detector (CPU/ONNX): {det_path}")
            detector = YOLO(str(det_path))
        else:
            print("[Model] WARNING: Stage 1 Detector not found!")

    if hailo_classifier is None:
        cls_path = cls_onnx if cls_onnx.exists() else cls_pt
        if cls_path.exists():
            print(f"[Model] Loading Stage 2 Classifier (CPU/ONNX): {cls_path}")
            classifier = YOLO(str(cls_path))
        else:
            print("[Model] WARNING: Stage 2 Classifier not found!")

def run_stage1_detector(img_bgr, conf_val=0.45, imgsz_val=640):
    """Run Stage 1 leaf localization on Hailo-8L NPU or CPU/ONNX."""
    global detector, hailo_detector
    h, w = img_bgr.shape[:2]
    boxes = []  # [(x1, y1, x2, y2, conf), ...]
    
    t_start = time.perf_counter()
    if hailo_detector is not None:
        try:
            raw = hailo_detector.infer(img_bgr)
            boxes = hailo_postprocess_detector(raw, w, h, conf_thresh=conf_val)
        except Exception as e:
            print(f"[Hailo] Stage 1 detector error: {e}")
    elif detector is not None:
        try:
            results = detector.predict(source=img_bgr, conf=conf_val, imgsz=imgsz_val, verbose=False)[0]
            for box in results.boxes[:10]:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                det_conf = float(box.conf[0].item())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 + 5 and y2 > y1 + 5:
                    boxes.append((x1, y1, x2, y2, det_conf))
        except Exception as e:
            print(f"[YOLO] Stage 1 detector error: {e}")
    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000
    return boxes, latency_ms

def run_stage2_classifier(crop_bgr, min_cls_conf=0.50):
    """Run Stage 2 disease classification on Hailo-8L NPU or CPU/ONNX."""
    global classifier, hailo_classifier
    import numpy as np
    if crop_bgr is None or crop_bgr.size == 0:
        return "Unknown", 0.0, {"Healthy": 0.0, "Other Disease": 0.0, "PLFD": 0.0}, 0.0

    t_start = time.perf_counter()
    predicted_class = "Unknown"
    conf = 0.0
    all_probs = {"Healthy": 0.0, "Other Disease": 0.0, "PLFD": 0.0}
    
    if hailo_classifier is not None:
        try:
            raw = hailo_classifier.infer(crop_bgr)
            for name, tensor in raw.items():
                arr = np.squeeze(tensor).astype(np.float32)
                if arr.ndim == 1 and len(arr) >= 3:
                    # Softmax normalization
                    exp_arr = np.exp(arr - np.max(arr))
                    probs = exp_arr / np.sum(exp_arr)
                    
                    top_idx = int(np.argmax(probs))
                    conf = float(probs[top_idx])
                    
                    # Filter out uncertain background crops (random guessing < 50%)
                    if conf >= min_cls_conf:
                        predicted_class = CLASS_NAMES.get(top_idx, f"Class{top_idx}")
                    else:
                        predicted_class = "Uncertain"
                        
                    all_probs = {
                        "Healthy": round(float(probs[0]), 4),
                        "Other Disease": round(float(probs[1]), 4),
                        "PLFD": round(float(probs[2]), 4)
                    }
                    break
        except Exception as e:
            print(f"[Hailo] Stage 2 classifier error: {e}")
    elif classifier is not None:
        try:
            results = classifier.predict(source=crop_bgr, verbose=False)[0]
            probs2 = results.probs
            names2 = results.names
            top_idx = int(probs2.top1)
            conf = float(probs2.top1conf)
            if conf >= min_cls_conf:
                predicted_class = names2[top_idx]
            else:
                predicted_class = "Uncertain"
            all_probs = {names2[j]: float(probs2.data[j].item()) for j in range(len(names2))}
        except Exception as e:
            print(f"[YOLO] Stage 2 classifier error: {e}")
    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000
    return predicted_class, conf, all_probs, latency_ms

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stats')
def stats():
    global detector, classifier, hailo_detector, hailo_classifier
    if hailo_detector is None and hailo_classifier is None and detector is None and classifier is None:
        load_models()
    stage1 = (hailo_detector is not None) or (detector is not None)
    stage2 = (hailo_classifier is not None) or (classifier is not None)
    is_hailo = (hailo_detector is not None) or (hailo_classifier is not None)
    return jsonify({
        "raw": {"PLFD": 0, "Healthy": 0, "Other Disease": 0},
        "models_exist": {
            "stage1": stage1,
            "stage2": stage2
        },
        "hailo": is_hailo
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/gps', methods=['GET'])
def get_gps_telemetry():
    with gps_lock:
        data = dict(latest_gps_data)
    if data.get("latitude") and data.get("longitude"):
        data["maps_url"] = f"https://www.google.com/maps?q={data['latitude']},{data['longitude']}"
    else:
        data["maps_url"] = None
    return jsonify(data)

@app.route('/api/field_trials/csv', methods=['GET'])
def download_field_trials_csv():
    if not FIELD_MASTER_CSV.exists():
        init_field_master_csv()
    return send_file(
        FIELD_MASTER_CSV,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"field_trials_master_log_{datetime.now().strftime('%Y%m%d')}.csv"
    )

@app.route('/api/field_trials/stats', methods=['GET'])
def get_field_trials_stats():
    total_records = 0
    class_counts = {"Healthy": 0, "Other Disease": 0, "PLFD": 0, "None": 0}
    with CSV_LOCK:
        if FIELD_MASTER_CSV.exists():
            try:
                with open(FIELD_MASTER_CSV, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        total_records += 1
                        p_cls = row.get("Stage2_Pred_Class", "None")
                        if p_cls in class_counts:
                            class_counts[p_cls] += 1
                        else:
                            class_counts[p_cls] = 1
            except Exception as e:
                print(f"[FieldLogger] Stats read error: {e}")
    return jsonify({
        "total_field_records": total_records,
        "class_distribution": class_counts,
        "csv_path": str(FIELD_MASTER_CSV)
    })

def resolve_request_location(req):
    loc_data = {
        "latitude": None,
        "longitude": None,
        "altitude_m": None,
        "source": "None",
        "status": "unavailable",
        "maps_url": None
    }
    
    with gps_lock:
        if latest_gps_data.get("status") == "locked" and latest_gps_data.get("latitude"):
            loc_data["latitude"] = latest_gps_data["latitude"]
            loc_data["longitude"] = latest_gps_data["longitude"]
            loc_data["altitude_m"] = latest_gps_data.get("altitude_m")
            loc_data["source"] = "Hardware NEO-M8N"
            loc_data["status"] = "locked"
    
    # Client-side HTML5 Geolocation fallback if hardware GPS is still acquiring fix
    if loc_data["latitude"] is None and 'latitude' in req.form and 'longitude' in req.form:
        try:
            c_lat = float(req.form['latitude'])
            c_lon = float(req.form['longitude'])
            if -90.0 <= c_lat <= 90.0 and -180.0 <= c_lon <= 180.0:
                loc_data["latitude"] = round(c_lat, 6)
                loc_data["longitude"] = round(c_lon, 6)
                loc_data["source"] = "Mobile Device GPS"
                loc_data["status"] = "locked"
        except (ValueError, TypeError):
            pass
            
    if loc_data["latitude"] is not None and loc_data["longitude"] is not None:
        loc_data["maps_url"] = f"https://www.google.com/maps?q={loc_data['latitude']},{loc_data['longitude']}"
        
    return loc_data

@app.route('/predict', methods=['POST'])
def predict():
    global detector, classifier, hailo_detector, hailo_classifier
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(filepath)
        
        # Load if not loaded
        if hailo_detector is None and hailo_classifier is None and detector is None and classifier is None:
            load_models()
            
        stage1_ready = (hailo_detector is not None) or (detector is not None)
        stage2_ready = (hailo_classifier is not None) or (classifier is not None)
        
        if not stage1_ready:
            return jsonify({"error": "Stage 1 detector model weights not found."}), 400
        if not stage2_ready:
            return jsonify({"error": "Stage 2 classifier model weights not found."}), 400
            
        # Resolve GPS Location (Dual-fallback: Hardware GNSS module or Mobile Geolocation)
        loc_data = resolve_request_location(request)

        # Clean up old crop files to save space
        try:
            for old_crop in app.config['UPLOAD_FOLDER'].glob("crop_*.jpg"):
                old_crop.unlink(missing_ok=True)
        except Exception as e:
            print(f"Error cleaning up old crops: {e}")

        try:
            t_total_start = time.perf_counter()
            img = cv2.imread(str(filepath))
            if img is None:
                return jsonify({"error": "Failed to read image"}), 400
            h_orig, w_orig, _ = img.shape
            
            # Get settings from request with default values
            conf_val = 0.15
            if 'conf' in request.form:
                try:
                    conf_val = float(request.form['conf'])
                except ValueError:
                    pass
            
            imgsz_val = 640
            if 'imgsz' in request.form:
                try:
                    imgsz_val = int(request.form['imgsz'])
                except ValueError:
                    pass

            # Prepare permanent field trial logging artifacts
            now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
            perm_raw_name = f"RAW_{ts_slug}_{uuid.uuid4().hex[:6]}_{filename}"
            perm_raw_path = FIELD_LOGS_RAW / perm_raw_name
            try:
                cv2.imwrite(str(perm_raw_path), img)
                saved_raw_rel = f"logs/field_trials/raw_frames/{perm_raw_name}"
            except Exception:
                saved_raw_rel = f"uploads/{filename}"

            session_id = request.form.get("session_id", f"SESS_{datetime.now().strftime('%Y%m%d')}")
            sample_tag = request.form.get("sample_tag_id", request.form.get("tag_id", "UNTAGGED"))
            tree_clone = request.form.get("tree_clone", "RRIM 600")
            canopy_lvl = request.form.get("canopy_level", "Middle")
            gt_class = request.form.get("ground_truth_class", request.form.get("ground_truth", "Unverified"))
            disease_sev = request.form.get("disease_severity", "None")
            light_cond = request.form.get("lighting_condition", "Natural_Canopy")
            lux_val = request.form.get("lux_level", "")
            dist_val = request.form.get("distance_meters", request.form.get("distance", "1.0"))
            angle_val = request.form.get("angle_degrees", "90")
            wind_cond = request.form.get("wind_condition", "Moderate")
            cam_sensor = request.form.get("camera_sensor", "Pi_Camera_3_Wide")

            with gps_lock:
                sat_count = latest_gps_data.get("satellites", 0)
                fix_stat = latest_gps_data.get("status", "unavailable")

            # Predict Stage 1
            boxes, det_latency = run_stage1_detector(img, conf_val=conf_val, imgsz_val=imgsz_val)
            
            if len(boxes) == 0:
                total_latency = (time.perf_counter() - t_total_start) * 1000
                append_field_trial_log({
                    "Timestamp": now_iso,
                    "Session_ID": session_id,
                    "Sample_Tag_ID": sample_tag,
                    "Tree_Clone": tree_clone,
                    "Canopy_Level": canopy_lvl,
                    "Ground_Truth_Class": gt_class,
                    "Disease_Severity": disease_sev,
                    "Lighting_Condition": light_cond,
                    "Lux_Level_lx": lux_val,
                    "Distance_meters": dist_val,
                    "Angle_degrees": angle_val,
                    "Wind_Condition": wind_cond,
                    "Camera_Sensor": cam_sensor,
                    "GPS_Latitude": loc_data.get("latitude") or "",
                    "GPS_Longitude": loc_data.get("longitude") or "",
                    "GPS_Altitude_m": loc_data.get("altitude_m") or "",
                    "GPS_Satellites": sat_count,
                    "GPS_Fix_Status": fix_stat,
                    "GPS_Source": loc_data.get("source") or "None",
                    "Stage1_Detected": False,
                    "Stage1_Confidence": 0.0,
                    "Stage1_BBox_Coords": "[]",
                    "Stage2_Pred_Class": "None",
                    "Stage2_Confidence": 0.0,
                    "Stage1_Latency_ms": round(det_latency, 1),
                    "Stage2_Latency_ms": 0.0,
                    "Total_Latency_ms": round(total_latency, 1),
                    "Raw_Image_Path": saved_raw_rel,
                    "Crop_Image_Path": ""
                })
                return jsonify({
                    "leaf_detected": False,
                    "leaves": [],
                    "image_url": f"/uploads/{filename}",
                    "location": loc_data,
                    "telemetry": {
                        "det_latency_ms": round(det_latency, 1),
                        "cls_latency_ms": 0.0,
                        "total_latency_ms": round(total_latency, 1)
                    }
                })
                
            leaves = []
            cls_latency_total = 0.0
            
            for i, (x1, y1, x2, y2, det_conf) in enumerate(boxes[:10]):
                cropped_leaf = img[y1:y2, x1:x2]
                if cropped_leaf.size == 0:
                    continue
                    
                # Save crop
                crop_filename = f"crop_{i}_{filepath.stem}.jpg"
                crop_path = app.config['UPLOAD_FOLDER'] / crop_filename
                cv2.imwrite(str(crop_path), cropped_leaf)
                crop_relative_path = f"/uploads/{crop_filename}"
                
                # Permanent crop save
                perm_crop_name = f"CROP_{ts_slug}_{uuid.uuid4().hex[:6]}_leaf{i+1}.jpg"
                perm_crop_path = FIELD_LOGS_CROPS / perm_crop_name
                try:
                    cv2.imwrite(str(perm_crop_path), cropped_leaf)
                    saved_crop_rel = f"logs/field_trials/crops/{perm_crop_name}"
                except Exception:
                    saved_crop_rel = crop_relative_path
                
                predicted_class, class_conf, all_probs, cls_lat = run_stage2_classifier(cropped_leaf)
                cls_latency_total += cls_lat
                
                append_field_trial_log({
                    "Timestamp": now_iso,
                    "Session_ID": session_id,
                    "Sample_Tag_ID": sample_tag,
                    "Tree_Clone": tree_clone,
                    "Canopy_Level": canopy_lvl,
                    "Ground_Truth_Class": gt_class,
                    "Disease_Severity": disease_sev,
                    "Lighting_Condition": light_cond,
                    "Lux_Level_lx": lux_val,
                    "Distance_meters": dist_val,
                    "Angle_degrees": angle_val,
                    "Wind_Condition": wind_cond,
                    "Camera_Sensor": cam_sensor,
                    "GPS_Latitude": loc_data.get("latitude") or "",
                    "GPS_Longitude": loc_data.get("longitude") or "",
                    "GPS_Altitude_m": loc_data.get("altitude_m") or "",
                    "GPS_Satellites": sat_count,
                    "GPS_Fix_Status": fix_stat,
                    "GPS_Source": loc_data.get("source") or "None",
                    "Stage1_Detected": True,
                    "Stage1_Confidence": round(float(det_conf), 4),
                    "Stage1_BBox_Coords": f"[{x1},{y1},{x2},{y2}]",
                    "Stage2_Pred_Class": predicted_class,
                    "Stage2_Confidence": round(float(class_conf), 4),
                    "Stage1_Latency_ms": round(det_latency, 1),
                    "Stage2_Latency_ms": round(cls_lat, 1),
                    "Total_Latency_ms": round((det_latency + cls_lat), 1),
                    "Raw_Image_Path": saved_raw_rel,
                    "Crop_Image_Path": saved_crop_rel
                })
                
                leaves.append({
                    "index": i + 1,
                    "class_id": 1,
                    "box": {
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "w": x2 - x1, "h": y2 - y1,
                        "confidence": det_conf
                    },
                    "crop_path": crop_relative_path,
                    "final_diagnosis": predicted_class,
                    "confidence": class_conf,
                    "probs": all_probs
                })

            t_total_end = time.perf_counter()
            total_latency = (t_total_end - t_total_start) * 1000
            
            return jsonify({
                "leaf_detected": len(leaves) > 0,
                "leaves": leaves,
                "image_url": f"/uploads/{filename}",
                "location": loc_data,
                "telemetry": {
                    "det_latency_ms": round(det_latency, 1),
                    "cls_latency_ms": round(cls_latency_total, 1),
                    "total_latency_ms": round(total_latency, 1)
                }
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Error running models: {str(e)}"}), 500

def draw_inference_on_frame(frame, conf_val=0.15, imgsz_val=640, box_thickness=2, hide_labels=False):
    global detector, classifier, hailo_detector, hailo_classifier
    global leaf_id_counter, tracked_leaves

    # Ensure models are loaded
    if hailo_detector is None and hailo_classifier is None and detector is None and classifier is None:
        load_models()

    h, w, _ = frame.shape
    
    # ── Stage 1: Detect leaf bounding boxes ──────────────────
    boxes, _ = run_stage1_detector(frame, conf_val=conf_val, imgsz_val=imgsz_val)
    
    # ── Stage 2: Classify candidate leaf crops ───────────────
    current_detections = []
    for (x1, y1, x2, y2, det_conf) in boxes[:10]:
        try:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            raw_pred, class_conf, all_probs, _ = run_stage2_classifier(crop)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            current_detections.append({
                "box": (x1, y1, x2, y2),
                "centroid": (cx, cy),
                "raw_pred": raw_pred,
                "conf": class_conf,
                "probs": all_probs
            })
        except Exception as e:
            print(f"[Inference] Stage 2 error: {e}")

    # ── Centroid Association & Spatial/Probability EMA Smoothing ──
    matched_ids = set()
    for det in current_detections:
        cx, cy = det["centroid"]
        x1, y1, x2, y2 = det["box"]
        best_match_id = None
        min_dist = 99999.0

        for tid, track in tracked_leaves.items():
            dist = math.sqrt((cx - track.cx)**2 + (cy - track.cy)**2)
            if dist < min_dist:
                min_dist = dist
                best_match_id = tid

        if best_match_id is not None and min_dist < 130:
            matched_ids.add(best_match_id)
            tracked_leaves[best_match_id].update(
                x1, y1, x2, y2, det["raw_pred"], det["conf"], det["probs"]
            )
        else:
            leaf_id_counter += 1
            tracked_leaves[leaf_id_counter] = LeafTrack(
                x1, y1, x2, y2, det["raw_pred"], det["conf"], det["probs"]
            )
            matched_ids.add(leaf_id_counter)

    # Clean up stale tracks (grace period of 6 frames prevents momentary dropout flicker)
    dead_ids = []
    for uid, track in tracked_leaves.items():
        if uid not in matched_ids:
            track.lost_frames += 1
            if track.lost_frames > 6:
                dead_ids.append(uid)
                
    for uid in dead_ids:
        del tracked_leaves[uid]

    # ── Draw Stabilized Boxes & Badges ──
    for uid, track in tracked_leaves.items():
        # Retain short ghosting only if active
        if track.lost_frames > 4:
            continue
            
        x1, y1 = int(track.x1), int(track.y1)
        x2, y2 = int(track.x2), int(track.y2)
        x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
        x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))

        if (x2 - x1) < 15 or (y2 - y1) < 15:
            continue

        label = track.label
        conf = track.conf

        if label == "Healthy":
            color = (34, 197, 94)      # Bright Green
            badge_bg = (20, 83, 45)
        elif label == "PLFD":
            color = (59, 130, 246)     # Red/Orange
            badge_bg = (30, 27, 75)
        elif label == "Other Disease":
            color = (0, 165, 255)     # Amber/Orange
            badge_bg = (20, 60, 120)
        else:
            color = (200, 200, 200)
            badge_bg = (50, 50, 50)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

        if not hide_labels:
            text = f"{label} {int(conf * 100)}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # If box is at the very top of the frame, render badge INSIDE the box below top border
            if y1 < th + 10:
                bx1 = x1
                by1 = y1
                bx2 = min(w, x1 + tw + 10)
                by2 = min(h, y1 + th + 8)
                text_pos = (bx1 + 5, by2 - 4)
            else:
                # Normal: render badge ABOVE the top border
                bx1 = x1
                by1 = max(0, y1 - th - 8)
                bx2 = min(w, x1 + tw + 10)
                by2 = y1
                text_pos = (bx1 + 5, by2 - 4)

            overlay = frame.copy()
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), badge_bg, -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1)
            cv2.putText(frame, text, text_pos, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return frame

pi_camera_fps = 0.0

def gen_pi_camera_frames(live_detect=False, conf_val=0.15, box_thickness=2, hide_labels=False, camera_id=None, target_fps=30):
    global pi_camera_fps
    frame_interval = 1.0 / max(1, min(60, target_fps))
    
    fps_smooth = 0.0
    alpha = 0.1
    
    while True:
        loop_start = time.perf_counter()
        
        with camera_lock:
            cam = get_camera(camera_id)
            if cam is None or not cam.isOpened():
                time.sleep(0.1)
                continue
            success, frame = read_camera_frame()
            
        if not success or frame is None:
            time.sleep(0.01)
            continue
            
        try:
            if live_detect:
                frame = draw_inference_on_frame(frame, conf_val=conf_val, box_thickness=box_thickness, hide_labels=hide_labels)
                
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            print(f"[Streaming Error] {e}")
            
        elapsed = time.perf_counter() - loop_start
        if elapsed > 0:
            current_fps = 1.0 / elapsed
            fps_smooth = current_fps if fps_smooth == 0.0 else (alpha * current_fps + (1.0 - alpha) * fps_smooth)
            pi_camera_fps = round(fps_smooth, 1)
            
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

@app.route('/pi_video_feed')
def pi_video_feed():
    live_param = request.args.get('live_detect', request.args.get('detect', 'false'))
    live_detect = str(live_param).lower() in ['true', '1', 'yes']
    conf_val = float(request.args.get('conf', 0.15))
    box_thickness = int(request.args.get('box_thickness', 2))
    hide_labels = str(request.args.get('hide_labels', 'false')).lower() in ['true', '1', 'yes']
    target_fps = int(request.args.get('fps', 30))
            
    camera_id = None
    if 'camera_id' in request.args:
        try:
            camera_id = int(request.args.get('camera_id'))
        except ValueError:
            pass
    
    return Response(gen_pi_camera_frames(live_detect, conf_val, box_thickness, hide_labels, camera_id, target_fps), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pi_status')
def pi_status():
    global pi_camera_fps
    return jsonify({
        "fps": pi_camera_fps
    })

@app.route('/close_pi_camera', methods=['POST'])
def close_pi_camera_route():
    with camera_lock:
        release_camera()
    return jsonify({"status": "Pi camera released"})

@app.route('/predict_pi_camera', methods=['POST'])
def predict_pi_camera():
    global detector, classifier, hailo_detector, hailo_classifier
    if hailo_detector is None and hailo_classifier is None and detector is None and classifier is None:
        load_models()
        
    stage1_ready = (hailo_detector is not None) or (detector is not None)
    stage2_ready = (hailo_classifier is not None) or (classifier is not None)
    
    if not stage1_ready:
        return jsonify({"error": "Stage 1 detector model weights not found."}), 400
    if not stage2_ready:
        return jsonify({"error": "Stage 2 classifier model weights not found."}), 400

    camera_id = None
    if 'camera_id' in request.form:
        try:
            camera_id = int(request.form['camera_id'])
        except ValueError:
            pass

    # Grab frame under lock
    with camera_lock:
        cam = get_camera(camera_id)
        if cam is None:
            return jsonify({"error": "Physical camera not detected or busy on the Raspberry Pi."}), 400
            
        success, frame = read_camera_frame()
        if not success or frame is None:
            return jsonify({"error": "Failed to capture frame from Pi camera."}), 500

    # Save to uploads directory with unique name
    filename = f"pi_capture_{uuid.uuid4().hex[:8]}.jpg"
    filepath = app.config['UPLOAD_FOLDER'] / filename
    cv2.imwrite(str(filepath), frame)
    
    # Run prediction pipeline
    try:
        loc_data = resolve_request_location(request)
        t_total_start = time.perf_counter()
        img = cv2.imread(str(filepath))
        if img is None:
            return jsonify({"error": "Failed to read image"}), 400
        h_orig, w_orig, _ = img.shape
        
        conf_val = 0.15
        if 'conf' in request.form:
            try:
                conf_val = float(request.form['conf'])
            except ValueError:
                pass
        
        imgsz_val = 640
        if 'imgsz' in request.form:
            try:
                imgsz_val = int(request.form['imgsz'])
            except ValueError:
                pass

        # Prepare permanent field trial logging artifacts
        now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        perm_raw_name = f"RAW_PICAM_{ts_slug}_{uuid.uuid4().hex[:6]}_{filename}"
        perm_raw_path = FIELD_LOGS_RAW / perm_raw_name
        try:
            cv2.imwrite(str(perm_raw_path), img)
            saved_raw_rel = f"logs/field_trials/raw_frames/{perm_raw_name}"
        except Exception:
            saved_raw_rel = f"uploads/{filename}"

        session_id = request.form.get("session_id", f"SESS_{datetime.now().strftime('%Y%m%d')}")
        sample_tag = request.form.get("sample_tag_id", request.form.get("tag_id", "UNTAGGED"))
        tree_clone = request.form.get("tree_clone", "RRIM 600")
        canopy_lvl = request.form.get("canopy_level", "Middle")
        gt_class = request.form.get("ground_truth_class", request.form.get("ground_truth", "Unverified"))
        disease_sev = request.form.get("disease_severity", "None")
        light_cond = request.form.get("lighting_condition", "Natural_Canopy")
        lux_val = request.form.get("lux_level", "")
        dist_val = request.form.get("distance_meters", request.form.get("distance", "1.0"))
        angle_val = request.form.get("angle_degrees", "90")
        wind_cond = request.form.get("wind_condition", "Moderate")
        cam_sensor = request.form.get("camera_sensor", "Pi_Camera_3_Wide")

        with gps_lock:
            sat_count = latest_gps_data.get("satellites", 0)
            fix_stat = latest_gps_data.get("status", "unavailable")

        boxes, det_latency = run_stage1_detector(img, conf_val=conf_val, imgsz_val=imgsz_val)

        if len(boxes) == 0:
            t_total_end = time.perf_counter()
            total_latency = (t_total_end - t_total_start) * 1000
            append_field_trial_log({
                "Timestamp": now_iso,
                "Session_ID": session_id,
                "Sample_Tag_ID": sample_tag,
                "Tree_Clone": tree_clone,
                "Canopy_Level": canopy_lvl,
                "Ground_Truth_Class": gt_class,
                "Disease_Severity": disease_sev,
                "Lighting_Condition": light_cond,
                "Lux_Level_lx": lux_val,
                "Distance_meters": dist_val,
                "Angle_degrees": angle_val,
                "Wind_Condition": wind_cond,
                "Camera_Sensor": cam_sensor,
                "GPS_Latitude": loc_data.get("latitude") or "",
                "GPS_Longitude": loc_data.get("longitude") or "",
                "GPS_Altitude_m": loc_data.get("altitude_m") or "",
                "GPS_Satellites": sat_count,
                "GPS_Fix_Status": fix_stat,
                "GPS_Source": loc_data.get("source") or "None",
                "Stage1_Detected": False,
                "Stage1_Confidence": 0.0,
                "Stage1_BBox_Coords": "[]",
                "Stage2_Pred_Class": "None",
                "Stage2_Confidence": 0.0,
                "Stage1_Latency_ms": round(det_latency, 1),
                "Stage2_Latency_ms": 0.0,
                "Total_Latency_ms": round(total_latency, 1),
                "Raw_Image_Path": saved_raw_rel,
                "Crop_Image_Path": ""
            })
            return jsonify({
                "leaf_detected": False,
                "leaves": [],
                "image_url": f"/uploads/{filename}",
                "location": loc_data,
                "telemetry": {
                    "det_latency_ms": round(det_latency, 1),
                    "cls_latency_ms": 0.0,
                    "total_latency_ms": round(total_latency, 1)
                }
            })
            
        leaves = []
        cls_latency_total = 0.0
        for i, (x1, y1, x2, y2, det_conf) in enumerate(boxes[:10]):
            cropped_leaf = img[y1:y2, x1:x2]
            if cropped_leaf.size == 0:
                continue
                
            crop_filename = f"crop_{i}_{filepath.stem}.jpg"
            crop_path = app.config['UPLOAD_FOLDER'] / crop_filename
            cv2.imwrite(str(crop_path), cropped_leaf)
            crop_relative_path = f"/uploads/{crop_filename}"
            
            # Permanent crop save
            perm_crop_name = f"CROP_PICAM_{ts_slug}_{uuid.uuid4().hex[:6]}_leaf{i+1}.jpg"
            perm_crop_path = FIELD_LOGS_CROPS / perm_crop_name
            try:
                cv2.imwrite(str(perm_crop_path), cropped_leaf)
                saved_crop_rel = f"logs/field_trials/crops/{perm_crop_name}"
            except Exception:
                saved_crop_rel = crop_relative_path
            
            predicted_class, class_conf, all_probs, cls_lat = run_stage2_classifier(cropped_leaf)
            cls_latency_total += cls_lat
            
            append_field_trial_log({
                "Timestamp": now_iso,
                "Session_ID": session_id,
                "Sample_Tag_ID": sample_tag,
                "Tree_Clone": tree_clone,
                "Canopy_Level": canopy_lvl,
                "Ground_Truth_Class": gt_class,
                "Disease_Severity": disease_sev,
                "Lighting_Condition": light_cond,
                "Lux_Level_lx": lux_val,
                "Distance_meters": dist_val,
                "Angle_degrees": angle_val,
                "Wind_Condition": wind_cond,
                "Camera_Sensor": cam_sensor,
                "GPS_Latitude": loc_data.get("latitude") or "",
                "GPS_Longitude": loc_data.get("longitude") or "",
                "GPS_Altitude_m": loc_data.get("altitude_m") or "",
                "GPS_Satellites": sat_count,
                "GPS_Fix_Status": fix_stat,
                "GPS_Source": loc_data.get("source") or "None",
                "Stage1_Detected": True,
                "Stage1_Confidence": round(float(det_conf), 4),
                "Stage1_BBox_Coords": f"[{x1},{y1},{x2},{y2}]",
                "Stage2_Pred_Class": predicted_class,
                "Stage2_Confidence": round(float(class_conf), 4),
                "Stage1_Latency_ms": round(det_latency, 1),
                "Stage2_Latency_ms": round(cls_lat, 1),
                "Total_Latency_ms": round((det_latency + cls_lat), 1),
                "Raw_Image_Path": saved_raw_rel,
                "Crop_Image_Path": saved_crop_rel
            })
            
            leaves.append({
                "index": i + 1,
                "class_id": 1,
                "box": {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "w": x2 - x1, "h": y2 - y1,
                    "confidence": det_conf
                },
                "crop_path": crop_relative_path,
                "final_diagnosis": predicted_class,
                "confidence": class_conf,
                "probs": all_probs
            })
                
        t_total_end = time.perf_counter()
        total_latency = (t_total_end - t_total_start) * 1000

        return jsonify({
            "leaf_detected": len(leaves) > 0,
            "leaves": leaves,
            "image_url": f"/uploads/{filename}",
            "location": loc_data,
            "telemetry": {
                "det_latency_ms": round(det_latency, 1),
                "cls_latency_ms": round(cls_latency_total, 1),
                "total_latency_ms": round(total_latency, 1)
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error running models on Pi capture: {str(e)}"}), 500


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run 2-Stage Rubber Leaf Detector Web App")
    parser.add_argument('--ssl', action='store_true', help="Enable ad-hoc SSL (HTTPS)")
    parser.add_argument('--host', default='0.0.0.0', help="Host address to bind to")
    parser.add_argument('--port', type=int, default=5000, help="Port to run on")
    args = parser.parse_args()

    load_models()
    
    ssl_context = None
    protocol = "http"
    if args.ssl:
        try:
            import OpenSSL
            ssl_context = 'adhoc'
            protocol = "https"
        except ImportError:
            print("pyOpenSSL not found. Falling back to HTTP.")

    print("\n" + "="*50)
    print("RUBBER LEAF DETECTOR SERVER STARTED!")
    print(f"Open {protocol}://127.0.0.1:{args.port} in your browser.")
    print("="*50 + "\n")

    app.run(host=args.host, port=args.port, debug=True, use_reloader=False, ssl_context=ssl_context)
