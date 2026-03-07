import time
from core.switcher import send_unicode_string
import subprocess
import ctypes

p = subprocess.Popen(['notepad.exe'])
time.sleep(1)

user32 = ctypes.windll.user32
hwnd = user32.FindWindowW("Notepad", None)
if hwnd:
    user32.SetForegroundWindow(hwnd)

time.sleep(0.5)

send_unicode_string("hello ")
time.sleep(0.5)
send_unicode_string("world")

time.sleep(2)
p.kill()
