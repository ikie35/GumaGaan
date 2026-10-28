#!/usr/bin/env bash
# ==============================================================================
#  Rubber Leaf AI Guard - Boot Tunnel Daemon Runner
# ==============================================================================
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL_FILE="$APP_DIR/tunnel_url.txt"
WEBHOOK_ENV="$APP_DIR/webhook.env"

echo "[BootTunnel] Starting Cloudflare Quick Tunnel to http://127.0.0.1:5000..."

# Wait until local Flask server is up before opening tunnel
until curl -s -o /dev/null http://127.0.0.1:5000/; do
    echo "[BootTunnel] Waiting for local rubber_disease.service to start..."
    sleep 2
done

echo "[BootTunnel] Flask app is responsive. Spawning tunnel..."

cloudflared tunnel --url http://127.0.0.1:5000 2>&1 | while read -r line; do
    echo "$line"
    if [[ "$line" =~ (https://[a-zA-Z0-9-]+\.trycloudflare\.com) ]]; then
        TUNNEL_URL="${BASH_REMATCH[1]}"
        echo "$TUNNEL_URL" > "$URL_FILE"
        chmod 666 "$URL_FILE" 2>/dev/null || true
        
        echo ""
        echo "=========================================================="
        echo "  🚀 BOOT TUNNEL IS ACTIVE:"
        echo "  $TUNNEL_URL"
        echo "=========================================================="
        echo ""

        # Optional: push to Discord webhook if configured in webhook.env
        if [ -f "$WEBHOOK_ENV" ]; then
            # shellcheck source=/dev/null
            source "$WEBHOOK_ENV"
            if [ -n "$DISCORD_WEBHOOK" ]; then
                curl -s -H "Content-Type: application/json" -X POST \
                    -d "{\"content\":\"🌿 **Rubber Leaf AI Guard Online!**\\nDashboard Link: $TUNNEL_URL\"}" \
                    "$DISCORD_WEBHOOK" > /dev/null || true
                echo "[BootTunnel] Notification sent to Discord!"
            fi
            if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
                curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
                    -d "chat_id=$TELEGRAM_CHAT_ID" \
                    -d "text=🌿 Rubber Leaf AI Guard is Online! Dashboard: $TUNNEL_URL" > /dev/null || true
                echo "[BootTunnel] Notification sent to Telegram!"
            fi
        fi
    fi
done
