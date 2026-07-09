# NAD Controller Automation

Small Windows automation tools for switching an NAD receiver between two input sources through a Broadlink RM4 device.

## What is included

- `nad_control.py`: sends the IR command for `video2` or `video3` to the NAD receiver.
- `NADSourceSwitch.ahk`: watches for screensaver and mouse activity, then calls the Python script to switch sources automatically.
- `monitor_logger.py`: diagnostic script that prints display changes for troubleshooting.

## Requirements

- Windows
- Python 3.10+ recommended
- A Broadlink RM4 Mini / RM4 Pro configured on your network
- AutoHotkey if you want to use the desktop automation script

## Installation

1. Create and activate a virtual environment if you want an isolated Python setup.
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Update the Broadlink connection settings in `nad_control.py`:

- `BROADLINK_IP`
- `BROADLINK_MAC`

## Usage

Run the Python script directly to switch the NAD receiver:

```bash
python nad_control.py video2
python nad_control.py video3
```

Source meanings in this project:

- `video2` = TV mode
- `video3` = PC mode

To use the automatic switcher, edit `NADSourceSwitch.ahk` so `PYTHON_PATH` and `SCRIPT_PATH` point to your local Python executable and this repository's `nad_control.py`, then run the script with AutoHotkey.

## Troubleshooting

- If the receiver does not switch, verify the Broadlink IP and MAC address in `nad_control.py`.
- If the AHK automation does nothing, confirm the paths in `NADSourceSwitch.ahk` are correct for your machine.
- Use `monitor_logger.py` to inspect display and monitor changes while testing source switching.

## Notes

- The IR codes are stored in `nad_control.py` as learned Pronto codes.
- `monitor_logger.py` uses PowerShell and Windows APIs, so it is intended for Windows only.