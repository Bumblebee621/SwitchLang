import time
from core.switcher import send_unicode_string
from core.keymap import char_to_vk
from core.switcher import send_keys
import subprocess
import ctypes

p = subprocess.Popen(['notepad.exe'])
time.sleep(1)

user32 = ctypes.windll.user32
hwnd = user32.FindWindowW("Notepad", None)
if hwnd:
    user32.SetForegroundWindow(hwnd)

time.sleep(0.5)

send_unicode_string("test1 ")
time.sleep(0.5)
send_unicode_string("test2 ")

time.sleep(0.5)

# Also try send_keys space
vks = []
en_vk, shifted = char_to_vk(' ')
if en_vk:
    vks.append((en_vk, shifted))
send_keys(vks)

time.sleep(1)
p.kill()
