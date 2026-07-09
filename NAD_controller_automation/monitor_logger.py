import subprocess
import time
import json
from datetime import datetime

def get_monitor_info():
    """Get current monitor state using PowerShell."""
    cmd = """
    Add-Type -AssemblyName System.Windows.Forms
    $screens = [System.Windows.Forms.Screen]::AllScreens
    $result = @()
    foreach ($screen in $screens) {
        $result += @{
            DeviceName = $screen.DeviceName
            Primary = $screen.Primary
            Bounds = "$($screen.Bounds.Width)x$($screen.Bounds.Height)"
            WorkingArea = "$($screen.WorkingArea.Width)x$($screen.WorkingArea.Height)"
        }
    }
    $result | ConvertTo-Json
    """
    result = subprocess.run(
        ["powershell", "-Command", cmd],
        capture_output=True, text=True
    )
    return result.stdout.strip()

def get_display_devices():
    """Get detailed display device info."""
    cmd = """
    Get-WmiObject -Class Win32_DesktopMonitor | Select-Object Name, DeviceID, Status, ScreenWidth, ScreenHeight | ConvertTo-Json
    """
    result = subprocess.run(
        ["powershell", "-Command", cmd],
        capture_output=True, text=True
    )
    return result.stdout.strip()

def get_display_config():
    """Get display configuration."""
    cmd = """
    Get-CimInstance -Namespace root/wmi -ClassName WmiMonitorBasicDisplayParams | Select-Object Active, InstanceName | ConvertTo-Json
    """
    result = subprocess.run(
        ["powershell", "-Command", cmd],
        capture_output=True, text=True
    )
    return result.stdout.strip()

print("=" * 60)
print("HDMI Monitor State Logger")
print("Switch your TV source while this runs")
print("Press Ctrl+C to stop")
print("=" * 60)

prev_monitors = None
prev_devices = None

while True:
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    monitors = get_monitor_info()
    devices = get_display_devices()
    config = get_display_config()
    
    if monitors != prev_monitors or devices != prev_devices:
        print(f"\n[{timestamp}] *** CHANGE DETECTED ***")
        print(f"--- Screens ---")
        print(monitors)
        print(f"--- Devices ---")
        print(devices)
        print(f"--- Config ---")
        print(config)
        prev_monitors = monitors
        prev_devices = devices
    else:
        print(f"[{timestamp}] No change - {monitors.count('DeviceName')} monitor(s) detected", end='\r')
    
    time.sleep(1)
