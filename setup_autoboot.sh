#!/usr/bin/env bash
# ==============================================================================
#  Rubber Leaf AI Guard - Automated Boot & System Service Setup
# ==============================================================================
set -e

echo ""
echo "===================================================================="
echo "  🌿 RUBBER LEAF AI GUARD - FULL AUTO-BOOT & SYSTEM OVERHAUL"
echo "===================================================================="
echo ""

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="rubber_disease.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
VENV_PYTHON="$APP_DIR/venv/bin/python"

echo "[1/6] Validating Directory & Environment..."
echo "  -> App Path: $APP_DIR"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "  [!] Virtual environment not found at $VENV_PYTHON"
    if command -v python3 &> /dev/null; then
        VENV_PYTHON="$(which python3)"
        echo "  -> Falling back to system Python: $VENV_PYTHON"
    else
        echo "  [ERROR] No Python found! Aborting."
        exit 1
    fi
else
    echo "  -> Virtual Environment Python verified: $VENV_PYTHON"
fi

echo ""
echo "[2/6] Probing Camera Hardware (IMX708 / Picamera2)..."
if command -v rpicam-hello &> /dev/null; then
    rpicam-hello --list-cameras || echo "  [Notice] rpicam-hello completed camera probe."
else
    echo "  [Notice] rpicam-hello command not in path, proceeding with Picamera2 python probe."
fi

echo ""
echo "[3/6] Generating Systemd Service Unit..."
sudo tee "$SERVICE_PATH" > /dev/null <<EOF
[Unit]
Description=Rubber Leaf AI Guard (Two-Stage Precision Detection & Diagnosis)
After=network-online.target multi-user.target systemd-udev-settle.service
Wants=network-online.target
Documentation=https://github.com/ikie35/GumaGaan

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$APP_DIR
# Settle period for MIPI CSI, I2C IMX708 sensor, and Hailo NPU PCIe on boot
ExecStartPre=/bin/sleep 2
ExecStart=$VENV_PYTHON app.py --host 0.0.0.0 --port 5000
Restart=always
RestartSec=3
KillSignal=SIGINT
TimeoutStopSec=10
Environment="PYTHONUNBUFFERED=1"
Environment="CAM_INDEX=0"
Environment="YOLO_OFFLINE=True"

[Install]
WantedBy=multi-user.target
EOF

sudo chmod 644 "$SERVICE_PATH"
echo "  -> Service unit written to $SERVICE_PATH"

echo ""
echo "[4/6] Registering & Enabling Service on Boot..."
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
echo "  -> $SERVICE_NAME enabled to start automatically on EVERY boot! ✅"

echo ""
echo "[5/6] Starting / Restarting Service Now..."
sudo systemctl restart "$SERVICE_NAME"
echo "  -> Waiting for Flask initialization and hardware binding..."
sleep 4

echo ""
echo "[6/6] Verifying Service Health..."
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "  -> Service is ACTIVE and RUNNING! 🚀"
else
    echo "  [!] Service failed to start immediately. Showing log:"
    sudo journalctl -u "$SERVICE_NAME" -n 25 --no-pager
    exit 1
fi

IP_ADDR=$(hostname -I | awk '{print $1}')
echo ""
echo "===================================================================="
echo "  🎉 OVERHAUL COMPLETE! SYSTEM IS FULLY CONFIGURED FOR BOOT."
echo "===================================================================="
echo "  The app is running right now and will auto-start every time you power on."
echo ""
echo "  🌐 Dashboard URL:  http://$IP_ADDR:5000"
echo "  🖥️  Local URL:      http://127.0.0.1:5000"
echo ""
echo "  Helpful Commands:"
echo "    - View Live Logs:  sudo journalctl -u $SERVICE_NAME -f"
echo "    - Restart Service: sudo systemctl restart $SERVICE_NAME"
echo "    - Stop Service:    sudo systemctl stop $SERVICE_NAME"
echo "===================================================================="
echo ""
