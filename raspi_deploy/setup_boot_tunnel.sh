#!/usr/bin/env bash
# ==============================================================================
#  Rubber Leaf AI Guard - Boot Tunnel Installer & Service Setup
#  Enables automatic Cloudflare tunneling on EVERY boot.
# ==============================================================================
set -e

echo ""
echo "===================================================================="
echo "  🌐 RUBBER LEAF AI GUARD - BOOT TUNNEL INSTALLER"
echo "===================================================================="
echo ""

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAFE_DIR="$HOME/rubber_guard"
ln -sfn "$APP_DIR" "$SAFE_DIR"

# 1. Install cloudflared if not present
if ! command -v cloudflared &> /dev/null; then
    echo "[1/4] Installing cloudflared binary for Raspberry Pi 5..."
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        DEB_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb"
    else
        DEB_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-armhf.deb"
    fi
    curl -fSL "$DEB_URL" -o /tmp/cloudflared.deb
    sudo dpkg -i /tmp/cloudflared.deb || sudo apt-get install -f -y
    rm -f /tmp/cloudflared.deb
    echo "  -> cloudflared installed successfully! ✅"
else
    echo "[1/4] cloudflared is already installed! ✅"
fi

# 2. Make runner executable
chmod +x "$SAFE_DIR/run_tunnel.sh"
echo "[2/4] Configured $SAFE_DIR/run_tunnel.sh ✅"

# 3. Install systemd service unit
echo "[3/4] Registering rubber_tunnel.service..."
SERVICE_FILE="/etc/systemd/system/rubber_tunnel.service"
sudo cp "$SAFE_DIR/rubber_tunnel.service" "$SERVICE_FILE"
sudo chmod 644 "$SERVICE_FILE"

sudo systemctl daemon-reload
sudo systemctl enable rubber_tunnel.service
echo "  -> rubber_tunnel.service enabled to start on EVERY boot! ✅"

# 4. Start service now and get the URL
echo "[4/4] Starting rubber_tunnel.service now..."
sudo systemctl restart rubber_tunnel.service
echo "  -> Waiting for Cloudflare connection handshake..."

URL_FILE="$SAFE_DIR/tunnel_url.txt"
rm -f "$URL_FILE"

FOUND_URL=""
for i in {1..20}; do
    sleep 1
    if [ -f "$URL_FILE" ] && [ -s "$URL_FILE" ]; then
        FOUND_URL=$(cat "$URL_FILE")
        break
    fi
done

echo ""
echo "===================================================================="
echo "  🎉 TUNNEL ON BOOT IS FULLY CONFIGURED & RUNNING!"
echo "===================================================================="
if [ -n "$FOUND_URL" ]; then
    echo "  📱 Access your dashboard on your phone using this link:"
    echo ""
    echo "     $FOUND_URL"
    echo ""
else
    echo "  Tunnel service is running in the background."
    echo "  Check the live URL at any time with:"
    echo "     cat $SAFE_DIR/tunnel_url.txt"
    echo "     sudo journalctl -u rubber_tunnel.service -n 20 --no-pager"
fi
echo "===================================================================="
echo "  From now on, whenever your Raspberry Pi powers on, it will"
echo "  automatically open this tunnel so you can access it from anywhere."
echo "===================================================================="
echo ""
