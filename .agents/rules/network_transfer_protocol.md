# Standard Protocol: Switching Raspberry Pi from Hotspot Mode to Wi-Fi

Every time the user or system transitions the Raspberry Pi 5 from Hotspot (Access Point) mode back to connecting to a standard Wi-Fi network:

### Primary Procedure:
1. **Desktop GUI / VNC / Screen:**
   - Click the Wi-Fi icon located in the top-right desktop panel tray.
   - If Hotspot mode is active, toggle/turn off the Hotspot.
   - Select the target Wi-Fi SSID from the dropdown list.
   - Enter the Wi-Fi password and confirm connection.

2. **Verify Assigned IP Address:**
   - Open terminal on the Pi and run:
     ```bash
     hostname -I
     ```
   - Note the new IP (e.g., `192.168.1.xxx` or `192.168.137.xxx`).

3. **Deploy / Connect from Windows:**
   - Run the upload script with the newly assigned IP address:
     ```powershell
     .\upload_to_raspi.ps1 -TargetIP "<NEW_PI_IP>"
     ```
