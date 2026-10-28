#!/usr/bin/env bash
# ==============================================================================
#  Rubber Leaf AI Guard - Zero-Lag Field Wi-Fi Hotspot Engine
#  Creates a direct peer-to-peer Wi-Fi network between the Pi and your Phone.
#  Latency: ~5ms (Instant, real-time camera aiming with zero lag).
# ==============================================================================
set -e

echo ""
echo "===================================================================="
echo "  🌿 STARTING ZERO-LAG FIELD WI-FI (DIRECT LOCAL CONNECTION)"
echo "===================================================================="
echo ""

HOTSPOT_NAME="RubberLeaf-AI-Net"
HOTSPOT_PASS="rubberleaf123"

echo "[1/4] Cleaning previous Wi-Fi connections..."
sudo nmcli connection down "$HOTSPOT_NAME" 2>/dev/null || true
sudo nmcli connection down Hotspot 2>/dev/null || true

echo "[2/4] Starting Direct Field Wi-Fi: $HOTSPOT_NAME..."
sudo nmcli device wifi hotspot ssid "$HOTSPOT_NAME" password "$HOTSPOT_PASS" ifname wlan0

echo "[3/4] Configuring Firewall Rules for Direct Access..."
# Allow incoming traffic on port 5000 and standard HTTP port 80
sudo iptables -I INPUT 1 -p tcp --dport 5000 -j ACCEPT 2>/dev/null || true
sudo iptables -I INPUT 1 -p tcp --dport 80 -j ACCEPT 2>/dev/null || true

# Port forward port 80 to 5000 so users can type just http://10.42.0.1 without :5000
sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 5000 2>/dev/null || true

echo "[4/4] Restarting Rubber Leaf AI Service on Direct Interface..."
sudo systemctl restart rubber_disease
sleep 2

echo ""
echo "===================================================================="
echo "  🚀 FIELD WI-FI IS ACTIVE! ZERO-LAG REAL-TIME POINTING READY"
echo "===================================================================="
echo ""
echo "  1. On your phone: Turn OFF Mobile Data (cellular)."
echo "  2. Go to Wi-Fi settings and connect to:"
echo "       SSID:     $HOTSPOT_NAME"
echo "       Password: $HOTSPOT_PASS"
echo "  3. Open Chrome on your phone and go to:"
echo "       http://10.42.0.1"
echo "       (or http://10.42.0.1:5000)"
echo ""
echo "  Latency will be ~5 milliseconds — instant, zero-lag camera aiming!"
echo "===================================================================="
echo ""
