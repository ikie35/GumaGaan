#!/usr/bin/env python3
"""
================================================================================
Raspberry Pi Camera Capture & Two-Stage Rubber Leaf Disease Diagnostic Tool
================================================================================
Supports Raspberry Pi Camera Module 3 (IMX708 PDAF), Module 2, HQ Camera, and USB cams.
Auto-detects the best backend:
  1. Picamera2 (Native Raspberry Pi OS Bookworm camera stack with hardware autofocus)
  2. OpenCV VideoCapture (via libcamerify or V4L2)
  3. rpicam-still / libcamera-still (CLI fallback)

Usage:
  - Single snapshot:
      python raspi_capture.py
      python raspi_capture.py --output my_leaf.jpg

  - Capture and immediately run 2-Stage AI Diagnosis (Stage 1 Detector + Stage 2 Classifier):
      python raspi_capture.py --predict
      python raspi_capture.py --predict --conf 0.25

  - Interactive live preview (press SPACE to snap, P to predict, A to autofocus, Q to quit):
      python raspi_capture.py --interactive

  - Burst / Timelapse capture (for field data collection):
      python raspi_capture.py --interval 3 --count 10 --dir ./field_captures

  - Hardware camera diagnostic check:
      python raspi_capture.py --diagnostics
================================================================================
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output encoding for terminals (Windows / Linux)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

# Force offline mode for ultralytics if used
os.environ["YOLO_OFFLINE"] = "True"

SCRIPT_DIR = Path(__file__).resolve().parent
CAPTURES_DIR = SCRIPT_DIR / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "Healthy", 1: "Other Disease", 2: "PLFD"}

class PiCameraHandler:
    """Unified camera controller that seamlessly interfaces with Picamera2 or OpenCV."""
    def __init__(self, width=1920, height=1080, camera_index=0):
        self.width = width
        self.height = height
        self.camera_index = camera_index
        self.backend = None
        self.picam2 = None
        self.cap = None
        self._init_camera()

    def _init_camera(self):
        # 1. Try native Picamera2 (Raspberry Pi OS Bookworm default)
        try:
            from picamera2 import Picamera2
            print("[Camera] Attempting native Picamera2 initialization...")
            p2 = Picamera2()
            config = p2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "BGR888"}
            )
            p2.configure(config)
            
            # Enable autofocus for Pi Camera Module 3 if supported
            try:
                # AfMode 2 = Continuous Autofocus
                p2.set_controls({"AfMode": 2})
                print("[Camera] Continuous Autofocus (PDAF) enabled for Pi Camera 3.")
            except Exception:
                pass
                
            p2.start()
            # Allow sensor auto-exposure and auto-white-balance to settle
            time.sleep(1.0)
            self.picam2 = p2
            self.backend = "picamera2"
            print("[Camera] Picamera2 backend initialized successfully! ✅")
            return
        except Exception as e:
            print(f"[Camera] Picamera2 unavailable or failed ({e}).")

        # 2. Try OpenCV VideoCapture (works with libcamerify or V4L2)
        try:
            print(f"[Camera] Attempting OpenCV VideoCapture (index={self.camera_index})...")
            cap = cv2.VideoCapture(self.camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    self.backend = "opencv"
                    print("[Camera] OpenCV VideoCapture backend initialized successfully! ✅")
                    return
                else:
                    cap.release()
        except Exception as e:
            print(f"[Camera] OpenCV VideoCapture failed ({e}).")

        # 3. If both failed, flag CLI fallback
        print("[Camera] Direct streaming backend not ready. Will fall back to rpicam-still CLI.")
        self.backend = "rpicam-cli"

    def read_frame(self):
        """Reads a single BGR frame from the active camera."""
        if self.backend == "picamera2" and self.picam2 is not None:
            try:
                frame = self.picam2.capture_array()
                return True, frame
            except Exception as e:
                print(f"[Camera Error] Picamera2 frame grab error: {e}")
                return False, None

        elif self.backend == "opencv" and self.cap is not None:
            ret, frame = self.cap.read()
            return ret, frame

        elif self.backend == "rpicam-cli":
            temp_file = CAPTURES_DIR / "_temp_snap.jpg"
            cmd = ["rpicam-still", "-t", "800", "--width", str(self.width), "--height", str(self.height), "-o", str(temp_file), "-n"]
            try:
                res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if res.returncode == 0 and temp_file.exists():
                    frame = cv2.imread(str(temp_file))
                    temp_file.unlink(missing_ok=True)
                    if frame is not None:
                        return True, frame
            except FileNotFoundError:
                # Try legacy libcamera-still
                cmd[0] = "libcamera-still"
                res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if res.returncode == 0 and temp_file.exists():
                    frame = cv2.imread(str(temp_file))
                    temp_file.unlink(missing_ok=True)
                    if frame is not None:
                        return True, frame

        return False, None

    def trigger_autofocus(self):
        """Manually trigger autofocus sweep on Pi Camera Module 3."""
        if self.backend == "picamera2" and self.picam2 is not None:
            try:
                # AfMode 1 = Auto (trigger single autofocus cycle)
                self.picam2.set_controls({"AfMode": 1, "AfTrigger": 0})
                time.sleep(0.1)
                self.picam2.set_controls({"AfTrigger": 1})
                print("[Camera] Triggered autofocus cycle.")
            except Exception as e:
                print(f"[Camera] Autofocus command error: {e}")

    def capture_image(self, save_path):
        """Captures a pristine, high-resolution photo to disk."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # In Picamera2, capture_file directly writes full-resolution JPEG with hardware ISP
        if self.backend == "picamera2" and self.picam2 is not None:
            try:
                self.picam2.capture_file(str(save_path))
                if save_path.exists() and save_path.stat().st_size > 0:
                    return True, save_path
            except Exception as e:
                print(f"[Camera] Picamera2 file capture failed ({e}), falling back to array write...")

        # Frame capture fallback
        success, frame = self.read_frame()
        if success and frame is not None:
            cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return True, save_path

        # CLI fallback
        if self.backend == "rpicam-cli":
            for tool in ["rpicam-still", "libcamera-still"]:
                try:
                    cmd = [tool, "-t", "1000", "--width", str(self.width), "--height", str(self.height), "-o", str(save_path), "-n", "--autofocus-mode", "auto"]
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    if res.returncode == 0 and save_path.exists():
                        return True, save_path
                except FileNotFoundError:
                    continue

        return False, None

    def release(self):
        """Releases camera hardware resources."""
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


# ─────────────────────────────────────────────────────────────
#  Two-Stage Model Inference Engine for CLI
# ─────────────────────────────────────────────────────────────
class TwoStageInference:
    def __init__(self):
        self.stage1_model = None
        self.stage2_model = None
        self.hailo_detector = None
        self.hailo_classifier = None
        self.mode = "Not loaded"
        self._load_models()

    def _load_models(self):
        models_dir = SCRIPT_DIR / "models"
        raspi_models_dir = SCRIPT_DIR / "raspi_deploy" / "models"

        det_hef = models_dir / "stage1_detector.hef"
        if not det_hef.exists():
            det_hef = raspi_models_dir / "stage1_detector.hef"
        if not det_hef.exists():
            det_hef = SCRIPT_DIR / "Stage 1 Rubber Leaf Detector" / "HEF models" / "stage1_single_class_yolov8n_best_hailo_model" / "stage1_single_class_yolov8n_best.hef"

        cls_hef = models_dir / "stage2_classifier.hef"
        if not cls_hef.exists():
            cls_hef = raspi_models_dir / "stage2_classifier.hef"
        if not cls_hef.exists():
            cls_hef = SCRIPT_DIR / "Stage2_Classifier" / "yolo11n-clsHEF" / "yolo11n-cls_best_hailo_model" / "yolo11n_robust_best.hef"

        # Try Hailo-8L NPU
        if det_hef.exists() and cls_hef.exists():
            try:
                from hailo_platform import (HEF, VDevice, HailoSchedulingAlgorithm,
                                             HailoStreamInterface, InferVStreams,
                                             ConfigureParams, InputVStreamParams,
                                             OutputVStreamParams, FormatType)
                params = VDevice.create_params()
                params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
                vdevice = VDevice(params)

                class SimpleHailo:
                    def __init__(self, hef_path, target):
                        self.hef = HEF(str(hef_path))
                        c_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
                        self.net_group = target.configure(self.hef, c_params)[0]
                        inp_p = InputVStreamParams.make(self.net_group, format_type=FormatType.UINT8)
                        out_p = OutputVStreamParams.make(self.net_group, format_type=FormatType.FLOAT32)
                        info = self.hef.get_input_vstream_infos()[0]
                        self.input_name = info.name
                        self.input_h = info.shape[0]
                        self.input_w = info.shape[1]
                        self.pipeline = InferVStreams(self.net_group, inp_p, out_p)
                        self.pipeline.__enter__()

                    def infer(self, img_bgr):
                        resized = cv2.resize(img_bgr, (self.input_w, self.input_h))
                        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                        batch = np.expand_dims(rgb, axis=0)
                        return self.pipeline.infer({self.input_name: batch})

                self.hailo_detector = SimpleHailo(det_hef, vdevice)
                self.hailo_classifier = SimpleHailo(cls_hef, vdevice)
                self.mode = "Hailo-8L NPU"
                print(f"[Model] Two-stage pipeline loaded on Hailo-8L NPU! ✅")
                return
            except Exception as e:
                print(f"[Model] Hailo-8L NPU load failed ({e}), checking PyTorch/ONNX...")

        # Fallback to PyTorch or ONNX via Ultralytics
        try:
            from ultralytics import YOLO
            det_pt = SCRIPT_DIR / "Stage 1 Rubber Leaf Detector" / "stage1_training_results" / "weights" / "stage1_best.pt"
            if not det_pt.exists():
                det_pt = SCRIPT_DIR / "Stage 1 Rubber Leaf Detector" / "stage1_training_results" / "weights" / "stage1_best.onnx"

            cls_pt = SCRIPT_DIR / "Stage2_Classifier" / "stage2_yolo11n_robust_results" / "best_weights" / "yolo11n_robust_best.pt"
            if not cls_pt.exists():
                cls_pt = SCRIPT_DIR / "Stage2_Classifier" / "Training Result on v4 dataset" / "weights" / "best.pt"
            if not cls_pt.exists():
                cls_pt = SCRIPT_DIR / "Stage2_Classifier" / "Training Result on v4 dataset" / "weights" / "best.onnx"

            if det_pt.exists() and cls_pt.exists():
                print(f"[Model] Loading Stage 1 Detector: {det_pt.name}")
                self.stage1_model = YOLO(str(det_pt))
                print(f"[Model] Loading Stage 2 Classifier: {cls_pt.name}")
                self.stage2_model = YOLO(str(cls_pt))
                self.mode = "Ultralytics YOLO (CPU/GPU)"
                print(f"[Model] Two-stage pipeline loaded via {self.mode}! ✅")
            else:
                print(f"[Model] WARNING: Model weights not found in standard paths.")
        except Exception as e:
            print(f"[Model] Ultralytics loading failed: {e}")

    def predict(self, img_bgr, conf_thresh=0.25):
        """Runs Stage 1 detection + Stage 2 classification, returns annotated image and detections list."""
        h, w = img_bgr.shape[:2]
        boxes = []
        detections = []
        t0 = time.perf_counter()

        # Stage 1 Detector
        if self.stage1_model is not None:
            res = self.stage1_model.predict(source=img_bgr, conf=conf_thresh, imgsz=640, verbose=False)[0]
            for box in res.boxes[:10]:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].item())
                boxes.append((x1, y1, x2, y2, conf))
        elif self.hailo_detector is not None:
            # Hailo NMS processing
            raw = self.hailo_detector.infer(img_bgr)
            for name, tensor in raw.items():
                arr = np.array(tensor).reshape(-1, 5)
                for det in arr:
                    conf = float(det[4])
                    if conf >= conf_thresh:
                        ymin, xmin, ymax, xmax = float(det[0]), float(det[1]), float(det[2]), float(det[3])
                        x1, y1 = int(xmin * w), int(ymin * h)
                        x2, y2 = int(xmax * w), int(ymax * h)
                        boxes.append((x1, y1, x2, y2, conf))

        t_det = (time.perf_counter() - t0) * 1000

        # Stage 2 Classifier on crops
        annotated = img_bgr.copy()
        for i, (x1, y1, x2, y2, det_conf) in enumerate(boxes):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            pred_label = "Unknown"
            cls_conf = 0.0

            if self.stage2_model is not None:
                c_res = self.stage2_model.predict(source=crop, verbose=False)[0]
                top1 = int(c_res.probs.top1)
                cls_conf = float(c_res.probs.top1conf)
                pred_label = c_res.names[top1]
            elif self.hailo_classifier is not None:
                c_raw = self.hailo_classifier.infer(crop)
                for name, tensor in c_raw.items():
                    arr = np.squeeze(tensor).astype(np.float32)
                    if arr.ndim == 1 and len(arr) >= 3:
                        exp_arr = np.exp(arr - np.max(arr))
                        probs = exp_arr / np.sum(exp_arr)
                        top1 = int(np.argmax(probs))
                        cls_conf = float(probs[top1])
                        pred_label = CLASS_NAMES.get(top1, f"Class{top1}")
                        break

            detections.append({
                "leaf_index": i + 1,
                "box": [x1, y1, x2, y2],
                "det_confidence": round(det_conf, 3),
                "diagnosis": pred_label,
                "cls_confidence": round(cls_conf, 3)
            })

            # Visual color scheme
            if pred_label == "Healthy":
                color = (0, 220, 0)       # Green
            elif pred_label == "PLFD":
                color = (0, 0, 255)       # Red
            else:
                color = (0, 165, 255)     # Amber / Other

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label_text = f"#{i+1} {pred_label} ({int(cls_conf*100)}%)"
            cv2.putText(annotated, label_text, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        total_latency = (time.perf_counter() - t0) * 1000
        return annotated, detections, total_latency


# ─────────────────────────────────────────────────────────────
#  Hardware Diagnostics Routine
# ─────────────────────────────────────────────────────────────
def run_diagnostics():
    print("=" * 70)
    print("  RASPBERRY PI CAMERA & MODEL HARDWARE DIAGNOSTIC REPORT")
    print("=" * 70)

    # 1. Check OS and Architecture
    print(f"Platform / OS : {sys.platform}")
    print(f"Python        : {sys.version.split()[0]}")

    # 2. Check Picamera2
    try:
        from picamera2 import Picamera2
        print("Picamera2     : INSTALLED ✅")
    except ImportError:
        print("Picamera2     : NOT INSTALLED (standard on Raspberry Pi OS Bookworm) ⚠️")

    # 3. Check V4L2 and rpicam tools
    v4l2_devices = list(Path("/dev").glob("video*"))
    print(f"V4L2 Devices  : {[str(d) for d in v4l2_devices] if v4l2_devices else 'None detected'}")

    rpicam_found = False
    for tool in ["rpicam-hello", "libcamera-hello"]:
        try:
            res = subprocess.run([tool, "--list-cameras"], capture_output=True, text=True, timeout=5)
            print(f"{tool} : Available ✅")
            if res.stdout:
                print("  Cameras detected by libcamera stack:")
                for line in res.stdout.strip().splitlines():
                    print(f"    {line}")
            rpicam_found = True
            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    if not rpicam_found:
        print("libcamera-apps: Neither 'rpicam-hello' nor 'libcamera-hello' found in PATH.")

    # 4. Check Hailo-8L NPU
    try:
        import hailo_platform
        print("HailoRT       : INSTALLED ✅")
    except ImportError:
        print("HailoRT       : Not installed in this Python environment.")

    # 5. Check Model files
    models_found = []
    for ext in ["*.hef", "*.onnx", "*.pt"]:
        for f in SCRIPT_DIR.glob(f"**/{ext}"):
            if ".git" not in str(f) and "__pycache__" not in str(f):
                models_found.append((f.name, f.stat().st_size / (1024*1024), str(f.relative_to(SCRIPT_DIR))))

    print(f"\nTrained Model Weights Discovered ({len(models_found)} total):")
    for name, size_mb, rel_path in sorted(models_found, key=lambda x: x[0]):
        print(f"  • {name:<32} ({size_mb:5.1f} MB) -> {rel_path}")

    print("=" * 70)


# ─────────────────────────────────────────────────────────────
#  Main CLI Logic
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Raspberry Pi Camera Capture & 2-Stage Rubber Leaf Disease Diagnostic Tool"
    )
    parser.add_argument("-o", "--output", type=str, default="", help="Output image file path (default: captures/capture_<timestamp>.jpg)")
    parser.add_argument("--predict", action="store_true", help="Immediately run 2-Stage leaf detector & disease classification on capture")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive live preview mode (SPACE to snap, P to predict, Q to quit)")
    parser.add_argument("--interval", type=float, default=0, help="Continuous capture interval in seconds (0 = single shot)")
    parser.add_argument("--count", type=int, default=1, help="Number of photos to capture in interval mode")
    parser.add_argument("--dir", type=str, default="captures", help="Output directory for captures")
    parser.add_argument("--width", type=int, default=1920, help="Image capture width (e.g. 1920, 1280)")
    parser.add_argument("--height", type=int, default=1080, help="Image capture height (e.g. 1080, 720)")
    parser.add_argument("--cam-index", type=int, default=0, help="V4L2 camera index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for Stage 1 leaf detection")
    parser.add_argument("--diagnostics", action="store_true", help="Run hardware camera and model diagnostic check")

    args = parser.parse_args()

    if args.diagnostics:
        run_diagnostics()
        return

    output_dir = Path(args.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Init] Starting Pi Camera Handler...")
    cam = PiCameraHandler(width=args.width, height=args.height, camera_index=args.cam_index)

    model_engine = None
    if args.predict or args.interactive:
        print("[Init] Initializing Two-Stage Inference Engine...")
        model_engine = TwoStageInference()

    try:
        # MODE 1: Interactive Live Preview
        if args.interactive:
            print("\n" + "=" * 60)
            print("  INTERACTIVE PI CAMERA VIEWFINDER")
            print("=" * 60)
            print("  [SPACE] / [C] : Capture photo to disk")
            print("  [P]           : Capture & run 2-Stage Leaf Diagnosis")
            print("  [A]           : Trigger Camera Module 3 Autofocus sweep")
            print("  [Q] / [ESC]   : Exit")
            print("=" * 60 + "\n")

            win_name = "Pi Camera Live Feed (Press Q to quit)"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 960, 540)

            while True:
                success, frame = cam.read_frame()
                if not success or frame is None:
                    time.sleep(0.02)
                    continue

                display_frame = frame.copy()
                cv2.putText(display_frame, "LIVE [SPACE: Snap | P: Predict | A: Focus | Q: Quit]",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(win_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    break
                elif key in [ord(' '), ord('c'), ord('C')]:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = output_dir / f"pi_capture_{ts}.jpg"
                    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    print(f"📸 Saved image: {out_path.name}")
                elif key in [ord('p'), ord('P')]:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    raw_path = output_dir / f"pi_raw_{ts}.jpg"
                    annotated_path = output_dir / f"pi_annotated_{ts}.jpg"
                    cv2.imwrite(str(raw_path), frame)
                    
                    if model_engine is not None:
                        annotated, dets, lat = model_engine.predict(frame, conf_thresh=args.conf)
                        cv2.imwrite(str(annotated_path), annotated)
                        print(f"🔬 Diagnosis: Found {len(dets)} leaf candidates in {lat:.1f}ms")
                        for d in dets:
                            print(f"   • Leaf #{d['leaf_index']}: {d['diagnosis']} ({d['cls_confidence']*100:.1f}%) [box: {d['box']}]")
                        print(f"   Annotated saved to: {annotated_path.name}")
                elif key in [ord('a'), ord('A')]:
                    cam.trigger_autofocus()

            cv2.destroyAllWindows()
            return

        # MODE 2: Interval / Burst Capture
        if args.interval > 0:
            print(f"[Burst] Capturing {args.count} photos every {args.interval}s into '{output_dir}'...")
            for i in range(args.count):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = output_dir / f"burst_{ts}_{i+1:03d}.jpg"
                ok, path = cam.capture_image(save_path)
                if ok:
                    print(f"[{i+1}/{args.count}] Captured: {path.name}")
                    if args.predict and model_engine is not None:
                        img = cv2.imread(str(path))
                        annotated, dets, lat = model_engine.predict(img, conf_thresh=args.conf)
                        ann_path = output_dir / f"annotated_{path.name}"
                        cv2.imwrite(str(ann_path), annotated)
                        print(f"    -> AI: {len(dets)} leaves ({lat:.1f}ms)")
                else:
                    print(f"[{i+1}/{args.count}] Capture failed!")

                if i < args.count - 1:
                    time.sleep(args.interval)
            print("[Burst] Completed!")
            return

        # MODE 3: Single Shot Capture
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_path = Path(args.output) if args.output else output_dir / f"capture_{ts}.jpg"

        print(f"[Capture] Taking photo with {cam.backend}...")
        success, final_path = cam.capture_image(target_path)
        if not success or final_path is None or not final_path.exists():
            print(f"[Error] Failed to capture image from camera.")
            sys.exit(1)

        size_kb = final_path.stat().st_size / 1024
        print(f"📸 Image successfully captured!")
        print(f"   Saved to: {final_path.resolve()} ({size_kb:.1f} KB)")

        # Optional prediction
        if args.predict and model_engine is not None:
            print("[Inference] Running Two-Stage Rubber Leaf AI...")
            img = cv2.imread(str(final_path))
            annotated, dets, latency = model_engine.predict(img, conf_thresh=args.conf)
            ann_path = final_path.parent / f"annotated_{final_path.name}"
            cv2.imwrite(str(ann_path), annotated)

            print("-" * 50)
            print(f"  AI DIAGNOSTIC RESULTS ({latency:.1f} ms total latency):")
            print("-" * 50)
            if len(dets) == 0:
                print("  No rubber leaves detected matching threshold.")
            else:
                for d in dets:
                    print(f"  • Leaf #{d['leaf_index']}: {d['diagnosis']:<14} (Confidence: {d['cls_confidence']*100:.1f}%)")
                    print(f"    Bounding Box: {d['box']} (Det Confidence: {d['det_confidence']*100:.1f}%)")
            print("-" * 50)
            print(f"  Annotated result saved to: {ann_path.resolve()}")

    finally:
        cam.release()

if __name__ == "__main__":
    main()
