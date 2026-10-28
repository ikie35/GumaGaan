#!/usr/bin/env bash
# ==============================================================================
#  Rubber Leaf AI Guard - Instant Tunneling Setup (Cloudflare Quick Tunnel)
#  Provides a secure public HTTPS URL to access the dashboard from ANY phone
#  or network without caring about local IPs or hotspot restrictions.
# ==============================================================================
set -e

echo ""
echo "===================================================================="
echo "  🌐 RUBBER LEAF AI GUARD - SECURE TUNNELING SETUP"
echo "===================================================================="
echo ""

# 1. Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "[1/3] Downloading & installing cloudflared for Raspberry Pi 5..."
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        DEB_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb"
    else
        DEB_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-armhf.deb"
    fi
    curl -fSL "$DEB_URL" -o /tmp/cloudflared.deb
    sudo dpkg -i /tmp/cloudflared.deb || sudo apt-get install -f -y
    rm -f /tmp/cloudflared.deb
    echo "  -> cloudflared successfully installed! ✅"
else
    echo "[1/3] cloudflared is already installed! ✅"
fi

echo ""
echo "[2/3] Checking local dashboard service..."
if ! systemctl is-active --quiet rubber_disease.service; then
    echo "  -> Starting rubber_disease.service..."
    sudo systemctl start rubber_disease.service
fi

echo ""
echo "[3/3] Launching Cloudflare Tunnel on http://127.0.0.1:5000..."
echo "===================================================================="
echo "  Look for the link ending in '.trycloudflare.com' below!"
echo "  You can open that HTTPS link on your phone from ANYWHERE in the world."
echo "===================================================================="
echo ""

# Launch quick tunnel (Ctrl+C to stop)
cloudflared tunnel --url http://127.0.0.1:5000
