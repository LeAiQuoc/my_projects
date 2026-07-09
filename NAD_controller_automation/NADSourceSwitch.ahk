; NAD Source Auto-Switcher
; Screensaver activates = TV mode (Video 2)
; First mouse click after screensaver = PC mode (Video 3)

#Persistent
#SingleInstance Force

PYTHON_PATH := "C:\Users\johnl\OneDrive - Academedia\Programming\private_projects\AutoHotkey\.venv\Scripts\python.exe"
SCRIPT_PATH := "C:\Users\johnl\OneDrive - Academedia\Programming\private_projects\AutoHotkey\nad_control.py"

watchingTV := false

SetTimer, CheckScreensaver, 1000
return

CheckScreensaver:
    DllCall("SystemParametersInfo", UInt, 0x0072, UInt, 0, UIntP, ssActive, UInt, 0)
    
    if (ssActive && !watchingTV) {
        ; Screensaver activated = watching TV
        watchingTV := true
        RunWait, "%PYTHON_PATH%" "%SCRIPT_PATH%" video2,, Hide
        TrayTip, NAD Remote, Screensaver on -> Video 2 (TV), 2
    }
return

; First mouse click after screensaver = back at PC
~LButton::
    if (watchingTV) {
        watchingTV := false
        RunWait, "%PYTHON_PATH%" "%SCRIPT_PATH%" video3,, Hide
        TrayTip, NAD Remote, Click detected -> Video 3 (PC), 2
    }
return

~RButton::
    if (watchingTV) {
        watchingTV := false
        RunWait, "%PYTHON_PATH%" "%SCRIPT_PATH%" video3,, Hide
        TrayTip, NAD Remote, Click detected -> Video 3 (PC), 2
    }
return

; Manual overrides
^!2::
    watchingTV := true
    RunWait, "%PYTHON_PATH%" "%SCRIPT_PATH%" video2,, Hide
    TrayTip, NAD Remote, Manual -> Video 2 (TV), 2
return

^!3::
    watchingTV := false
    RunWait, "%PYTHON_PATH%" "%SCRIPT_PATH%" video3,, Hide
    TrayTip, NAD Remote, Manual -> Video 3 (PC), 2
return
