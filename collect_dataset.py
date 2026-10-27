#!/usr/bin/env python3
"""
================================================================================
Rubber Leaf Dataset Collection & Refinement Tool (Raspberry Pi & Field Use)
================================================================================
Designed for collecting real-world field images to fine-tune and refine:
  - Stage 1: Rubber Leaf Bounding Box Detector (YOLOv8n / YOLO11n)
             (Positive leaf scenes & background-only negative suppressor images)
  - Stage 2: Disease Classifier (Healthy vs. PLFD vs. Other Disease)

Features:
  - Direct Picamera2 PDAF autofocus & high-res capture on Raspberry Pi 5.
  - Interactive categorization into labeled folders.
  - Automatic dataset zipping for Roboflow or Kaggle upload.
  - Full metadata CSV logging (GPS, timestamp, resolution).

Usage:
  Interactive menu:
    python collect_dataset.py

  Single shot with class tag:
    python collect_dataset.py --stage 2 --class PLFD
    python collect_dataset.py --stage 1 --type background_negative

  Timed burst for canopy sweeps:
    python collect_dataset.py --stage 1 --interval 2 --count 15

  Zip collected images:
    python collect_dataset.py --zip
================================================================================
"""

import os
import sys
import time
import zipfile
import argparse
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output encoding for terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

# Import unified camera handler from raspi_capture if available
try:
    from raspi_capture import PiCameraHandler
except ImportError:
    # Embedded fallback camera handler
    class PiCameraHandler:
        def __init__(self, width=1920, height=1080, camera_index=0):
            self.width = width
            self.height = height
            self.camera_index = camera_index
            self.backend = None
            self.picam2 = None
            self.cap = None
            self._init_camera()

        def _init_camera(self):
            try:
                from picamera2 import Picamera2
                p2 = Picamera2()
                config = p2.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "BGR888"}
                )
                p2.configure(config)
                try:
                    p2.set_controls({"AfMode": 2})
                except Exception:
                    pass
                p2.start()
                time.sleep(1.0)
                self.picam2 = p2
                self.backend = "picamera2"
                return
            except Exception:
                pass

            try:
                cap = cv2.VideoCapture(self.camera_index)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                if cap.isOpened():
                    self.cap = cap
                    self.backend = "opencv"
            except Exception:
                pass

        def read_frame(self):
            if self.backend == "picamera2" and self.picam2 is not None:
                try:
                    return True, self.picam2.capture_array()
                except Exception:
                    return False, None
            elif self.backend == "opencv" and self.cap is not None:
                return self.cap.read()
            return False, None

        def capture_image(self, path):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if self.backend == "picamera2" and self.picam2 is not None:
                try:
                    self.picam2.capture_file(str(path))
                    if path.exists() and path.stat().st_size > 0:
                        return True, path
                except Exception:
                    pass
            ok, frame = self.read_frame()
            if ok and frame is not None:
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                return True, path
            return False, None

        def trigger_autofocus(self):
            if self.backend == "picamera2" and self.picam2 is not None:
                try:
                    self.picam2.set_controls({"AfMode": 1, "AfTrigger": 0})
                    time.sleep(0.1)
                    self.picam2.set_controls({"AfTrigger": 1})
                except Exception:
                    pass

        def release(self):
            if self.picam2 is not None:
                try:
                    self.picam2.stop()
                    self.picam2.close()
                except Exception:
                    pass
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_BASE = SCRIPT_DIR / "dataset_refinement"

# Standard folder hierarchy
STAGE1_LEAF_DIR   = DATASET_BASE / "stage1_detector" / "leaf_samples"
STAGE1_BG_DIR     = DATASET_BASE / "stage1_detector" / "background_negatives"
STAGE2_HEALTHY    = DATASET_BASE / "stage2_classifier" / "Healthy"
STAGE2_PLFD       = DATASET_BASE / "stage2_classifier" / "PLFD"
STAGE2_OTHER      = DATASET_BASE / "stage2_classifier" / "Other_Disease"

for d in [STAGE1_LEAF_DIR, STAGE1_BG_DIR, STAGE2_HEALTHY, STAGE2_PLFD, STAGE2_OTHER]:
    d.mkdir(parents=True, exist_ok=True)

CSV_LOG_PATH = DATASET_BASE / "collection_manifest.csv"

def init_manifest():
    if not CSV_LOG_PATH.exists() or CSV_LOG_PATH.stat().st_size == 0:
        with open(CSV_LOG_PATH, "w", encoding="utf-8") as f:
            f.write("Timestamp,Stage,Category,Filename,RelativePath,FileSizeBytes\n")

init_manifest()

def log_sample(stage, category, filepath):
    filepath = Path(filepath)
    size_bytes = filepath.stat().st_size if filepath.exists() else 0
    rel_path = filepath.relative_to(DATASET_BASE) if DATASET_BASE in filepath.parents else filepath.name
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f'"{now_str}","{stage}","{category}","{filepath.name}","{rel_path}",{size_bytes}\n')

def print_status_summary():
    print("\n" + "=" * 65)
    print("  CURRENT COLLECTED REFINEMENT DATASET SUMMARY")
    print("=" * 65)
    counts = {
        "Stage 1 Leaf Samples"       : len(list(STAGE1_LEAF_DIR.glob("*.jpg"))),
        "Stage 1 Background Negatives": len(list(STAGE1_BG_DIR.glob("*.jpg"))),
        "Stage 2 Healthy Crops"      : len(list(STAGE2_HEALTHY.glob("*.jpg"))),
        "Stage 2 PLFD Crops"         : len(list(STAGE2_PLFD.glob("*.jpg"))),
        "Stage 2 Other Disease Crops": len(list(STAGE2_OTHER.glob("*.jpg"))),
    }
    total = sum(counts.values())
    for label, count in counts.items():
        print(f"  • {label:<30}: {count:>4} images")
    print("-" * 65)
    print(f"  TOTAL COLLECTED SAMPLES       : {total:>4} images")
    print(f"  Storage Location              : {DATASET_BASE}")
    print("=" * 65 + "\n")

def create_dataset_zip():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = SCRIPT_DIR / f"rubber_leaf_refinement_dataset_{ts}.zip"
    print(f"[Export] Creating archive: {zip_path.name}...")
    
    file_count = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in DATASET_BASE.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(DATASET_BASE)
                zf.write(file, arcname)
                file_count += 1
                
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[Export] SUCCESS! Zipped {file_count} files ({size_mb:.2f} MB).")
    print(f"[Export] Saved to: {zip_path.resolve()}")
    print("[Export] You can now upload this zip directly to Kaggle or Roboflow.")
    return zip_path

def interactive_collector():
    print_status_summary()
    print("Select collection mode:")
    print("  [1] Stage 1: Leaf Detection Sample (Positive Bounding Box Scenes)")
    print("  [2] Stage 1: Background Negative (Canopy/Trunk/Sky/Ground without rubber leaves)")
    print("  [3] Stage 2: Healthy Leaf Sample")
    print("  [4] Stage 2: PLFD (Pestalotiopsis Leaf Fall Disease) Sample")
    print("  [5] Stage 2: Other Disease / Nutrient Deficiency Sample")
    print("  [6] Interactive Viewfinder Mode (Live Preview with Keyboard Hotkeys)")
    print("  [7] Export Dataset to ZIP (for Kaggle/Roboflow)")
    print("  [Q] Exit")
    
    choice = input("\nEnter choice [1-7 or Q]: ").strip()
    
    if choice in ['q', 'Q', '']:
        return False
        
    if choice == '7':
        create_dataset_zip()
        return True

    dest_folder = None
    stage_name = ""
    cat_name = ""
    
    if choice == '1':
        dest_folder = STAGE1_LEAF_DIR
        stage_name = "Stage1"
        cat_name = "Leaf_Sample"
    elif choice == '2':
        dest_folder = STAGE1_BG_DIR
        stage_name = "Stage1"
        cat_name = "Background_Negative"
    elif choice == '3':
        dest_folder = STAGE2_HEALTHY
        stage_name = "Stage2"
        cat_name = "Healthy"
    elif choice == '4':
        dest_folder = STAGE2_PLFD
        stage_name = "Stage2"
        cat_name = "PLFD"
    elif choice == '5':
        dest_folder = STAGE2_OTHER
        stage_name = "Stage2"
        cat_name = "Other_Disease"
    elif choice == '6':
        run_viewfinder_mode()
        return True
    else:
        print("[Error] Invalid selection.")
        return True

    cam = PiCameraHandler()
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{cat_name.lower()}_{ts}.jpg"
        save_path = dest_folder / filename
        print(f"[Camera] Capturing sample for {cat_name}...")
        ok, res_path = cam.capture_image(save_path)
        if ok and res_path.exists():
            log_sample(stage_name, cat_name, res_path)
            print(f"📸 SAVED: {res_path.name} ({res_path.stat().st_size / 1024:.1f} KB)")
        else:
            print("[Error] Camera capture failed. Ensure camera is attached.")
    finally:
        cam.release()
        
    return True

def run_viewfinder_mode():
    print("\n" + "=" * 65)
    print("  INTERACTIVE DATASET COLLECTION VIEWFINDER")
    print("=" * 65)
    print("  [1] Snap Stage 1: Leaf Sample")
    print("  [2] Snap Stage 1: Background Negative")
    print("  [3] Snap Stage 2: Healthy")
    print("  [4] Snap Stage 2: PLFD")
    print("  [5] Snap Stage 2: Other Disease")
    print("  [A] Trigger Autofocus sweep")
    print("  [Q] Exit")
    print("=" * 65 + "\n")
    
    cam = PiCameraHandler()
    win_name = "Dataset Collector (1-5 to classify, Q to quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 540)
    
    try:
        while True:
            ok, frame = cam.read_frame()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
                
            display = frame.copy()
            cv2.putText(display, "COLLECTOR: 1:S1 Leaf | 2:S1 Neg | 3:Healthy | 4:PLFD | 5:Other | Q:Quit",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(win_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
                
            mapping = {
                ord('1'): (STAGE1_LEAF_DIR, "Stage1", "Leaf_Sample"),
                ord('2'): (STAGE1_BG_DIR,   "Stage1", "Background_Negative"),
                ord('3'): (STAGE2_HEALTHY,   "Stage2", "Healthy"),
                ord('4'): (STAGE2_PLFD,      "Stage2", "PLFD"),
                ord('5'): (STAGE2_OTHER,     "Stage2", "Other_Disease"),
            }
            
            if key in mapping:
                folder, stage, cat = mapping[key]
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = folder / f"{cat.lower()}_{ts}.jpg"
                cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                log_sample(stage, cat, save_path)
                print(f"📸 [{stage} - {cat}] Saved: {save_path.name}")
            elif key in [ord('a'), ord('A')]:
                cam.trigger_autofocus()
                print("🔍 Autofocus triggered.")
    finally:
        cam.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Rubber Leaf Dataset Collection & Refinement Tool")
    parser.add_argument("--stage", type=int, choices=[1, 2], help="Target fine-tuning stage (1=Detector, 2=Classifier)")
    parser.add_argument("--type", type=str, default="leaf", choices=["leaf", "background_negative"], help="For Stage 1: 'leaf' or 'background_negative'")
    parser.add_argument("--class", dest="class_name", type=str, choices=["Healthy", "PLFD", "Other_Disease"], help="For Stage 2 class")
    parser.add_argument("--interval", type=float, default=0, help="Continuous burst interval in seconds")
    parser.add_argument("--count", type=int, default=1, help="Number of photos to capture")
    parser.add_argument("--zip", action="store_true", help="Zip collected images for Kaggle/Roboflow export")
    parser.add_argument("--status", action="store_true", help="Print collected dataset statistics")
    
    args = parser.parse_args()

    if args.status:
        print_status_summary()
        return

    if args.zip:
        create_dataset_zip()
        return

    if args.stage is None:
        while True:
            cont = interactive_collector()
            if not cont:
                break
        return

    # CLI automated mode
    dest_dir = None
    stage_str = f"Stage{args.stage}"
    cat_str = ""
    
    if args.stage == 1:
        if args.type == "background_negative":
            dest_dir = STAGE1_BG_DIR
            cat_str = "Background_Negative"
        else:
            dest_dir = STAGE1_LEAF_DIR
            cat_str = "Leaf_Sample"
    elif args.stage == 2:
        if not args.class_name:
            print("[Error] Stage 2 collection requires --class (Healthy, PLFD, or Other_Disease)")
            sys.exit(1)
        cat_str = args.class_name
        dest_dir = DATASET_BASE / "stage2_classifier" / args.class_name

    cam = PiCameraHandler()
    try:
        print(f"[Collector] Capturing {args.count} sample(s) for {stage_str} ({cat_str})...")
        for i in range(args.count):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = dest_dir / f"{cat_str.lower()}_{ts}_{i+1:03d}.jpg"
            ok, p = cam.capture_image(save_path)
            if ok:
                log_sample(stage_str, cat_str, p)
                print(f"[{i+1}/{args.count}] Saved: {p.name}")
            else:
                print(f"[{i+1}/{args.count}] Capture failed.")
            if i < args.count - 1 and args.interval > 0:
                time.sleep(args.interval)
    finally:
        cam.release()

if __name__ == "__main__":
    main()
