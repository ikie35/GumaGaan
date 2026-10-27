#!/usr/bin/env python3
"""
Publication-Quality Figure Generator for Edge Hardware Stress Test Results
Generates multi-panel telemetry plots for Master's thesis / journal paper.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_stress_test(csv_path="logs/stress_test/STRESS_20260910_152150_telemetry.csv", output_png="logs/stress_test/stress_test_publication_figures.png"):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} telemetry samples from {csv_path}")

    # Set publication aesthetic style
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=300)
    fig.suptitle("Raspberry Pi 5 + Hailo-8L NPU: 1-Hour Continuous Stress Test Profiling", fontsize=16, fontweight='bold', y=0.98)

    # 1. Thermal Stability Profile
    ax1 = axes[0, 0]
    ax1.plot(df['Elapsed_Min'], df['CPU_Temp_C'], color='#d9534f', linewidth=2.0, label='Raspberry Pi 5 CPU Temp (°C)')
    mean_temp = df['CPU_Temp_C'].mean()
    max_temp = df['CPU_Temp_C'].max()
    ax1.axhline(mean_temp, color='#d9534f', linestyle='--', alpha=0.7, label=f'Mean Temp: {mean_temp:.1f}°C')
    ax1.axhline(80.0, color='darkred', linestyle=':', linewidth=1.5, label='Thermal Throttle Limit (80°C)')
    ax1.set_title("A. Thermal Profile & Dissipation Curve", fontweight='bold')
    ax1.set_xlabel("Elapsed Time (Minutes)")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_ylim(40, 85)
    ax1.legend(loc='lower right', frameon=True)

    # 2. Pipeline Latency Breakdown
    ax2 = axes[0, 1]
    ax2.plot(df['Elapsed_Min'], df['Stage1_Latency_ms'], color='#0275d8', linewidth=1.8, label='Stage 1 Detector (YOLO)')
    ax2.plot(df['Elapsed_Min'], df['Stage2_Latency_ms'], color='#5cb85c', linewidth=1.8, label='Stage 2 Classifier (YOLO11n-cls)')
    ax2.plot(df['Elapsed_Min'], df['Total_Latency_ms'], color='#292b2c', linewidth=2.0, linestyle='-', label='Total End-to-End Latency')
    ax2.set_title("B. Inference Latency Stability Across Iterations", fontweight='bold')
    ax2.set_xlabel("Elapsed Time (Minutes)")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_ylim(0, 35)
    ax2.legend(loc='upper right', frameon=True)

    # 3. Sustained Operational Throughput (FPS)
    ax3 = axes[1, 0]
    ax3.plot(df['Elapsed_Min'], df['Instant_FPS'], color='#5bc0de', linewidth=1.5, alpha=0.7, label='Instantaneous FPS')
    ax3.plot(df['Elapsed_Min'], df['Average_FPS'], color='#0275d8', linewidth=2.2, label=f"Average Throughput: {df['Average_FPS'].iloc[-1]:.1f} FPS")
    ax3.set_title("C. Operational Frame Rate (Throughput)", fontweight='bold')
    ax3.set_xlabel("Elapsed Time (Minutes)")
    ax3.set_ylabel("Frames Per Second (FPS)")
    ax3.set_ylim(20, 60)
    ax3.legend(loc='lower right', frameon=True)

    # 4. Memory (RAM) Consumption & Leak Verification
    ax4 = axes[1, 1]
    ax4.plot(df['Elapsed_Min'], df['RAM_Used_MB'], color='#6f42c1', linewidth=2.0, label='System RAM Used (MB)')
    ax4.set_title("D. Memory Stability (Zero Memory Leak Verification)", fontweight='bold')
    ax4.set_xlabel("Elapsed Time (Minutes)")
    ax4.set_ylabel("RAM Used (MB)")
    ax4.set_ylim(800, 2000)
    ax4.legend(loc='upper right', frameon=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), bbox_inches='tight')
    plt.close()
    print(f"Publication-ready figure saved successfully -> {out_path}")

if __name__ == "__main__":
    plot_stress_test()
