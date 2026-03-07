import time
import ctypes
from ctypes import wintypes
import subprocess

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ('wVk', wintypes.WORD),
        ('wScan', wintypes.WORD),
        ('dwFlags', wintypes.DWORD),
        ('time', wintypes.DWORD),
        ('dwExtraInfo', ctypes.POINTER(ctypes.c_ulong)),
    ]

class INPUT(ctypes.Structure):
    class _INPUT_UNION(ctypes.Union):
        _fields_ = [('ki', KEYBDINPUT)]
    _fields_ = [('type', wintypes.DWORD), ('union', _INPUT_UNION)]

def press_key(vk):
    user32 = ctypes.windll.user32
    down = INPUT()
    down.type = INPUT_KEYBOARD
    down.union.ki.wVk = vk
    
    up = INPUT()
    up.type = INPUT_KEYBOARD
    up.union.ki.wVk = vk
    up.union.ki.dwFlags = KEYEVENTF_KEYUP
    
    inputs = (INPUT * 2)(down, up)
    user32.SendInput(2, inputs, ctypes.sizeof(INPUT))

print("Launching Notepad...")
p = subprocess.Popen(['notepad.exe'])
time.sleep(1)

user32 = ctypes.windll.user32
hwnd = user32.FindWindowW("Notepad", None)
if hwnd:
    user32.SetForegroundWindow(hwnd)

time.sleep(0.5)

# Type 'tueh ' + 'foo ' extremely fast
print("Typing 'tueh foo ' with 0 delay between keys...")
keys = [0x54, 0x55, 0x45, 0x48, 0x20, 0x46, 0x4F, 0x4F, 0x20]
for vk in keys:
    press_key(vk)

print("Check Notepad for 'אוקי כםם '.")
time.sleep(2)
p.kill()
