"""
switcher.py — Concurrency-safe keyboard layout correction sequence.

Handles the critical section: erasing wrong text, toggling the OS
keyboard layout, injecting corrected text, and flushing any keys
typed during the correction phase.
"""

import ctypes
import ctypes.wintypes as wintypes
import time

user32 = ctypes.windll.user32

# Ensure correct return types for 64-bit compatibility
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.GetKeyboardLayout.restype = wintypes.HKL

# Windows constants
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
VK_BACK = 0x08
WM_INPUTLANGCHANGEREQUEST = 0x0050

# Hebrew and English HKL handles
HKL_EN_US = 0x04090409
HKL_HE = 0x040D040D


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ('dx', wintypes.LONG),
        ('dy', wintypes.LONG),
        ('mouseData', wintypes.DWORD),
        ('dwFlags', wintypes.DWORD),
        ('time', wintypes.DWORD),
        ('dwExtraInfo', ctypes.POINTER(ctypes.c_ulong)),
    ]

# HARDWAREINPUT is smaller, omitting for brevity.

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
        _fields_ = [
            ('ki', KEYBDINPUT),
            ('mi', MOUSEINPUT),
        ]

    _fields_ = [
        ('type', wintypes.DWORD),
        ('union', _INPUT_UNION),
    ]


def _make_key_input(vk=0, scan=0, flags=0):
    """Create an INPUT structure for a keyboard event."""
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki.wVk = vk
    inp.union.ki.wScan = scan
    inp.union.ki.dwFlags = flags
    inp.union.ki.time = 0
    inp.union.ki.dwExtraInfo = ctypes.pointer(ctypes.c_ulong(0))
    return inp


def _send_inputs(inputs):
    """Send a list of INPUT structures to the OS."""
    n = len(inputs)
    arr = (INPUT * n)(*inputs)
    user32.SendInput(n, arr, ctypes.sizeof(INPUT))


def send_backspaces(count):
    """Send N backspace key events to erase characters.

    Args:
        count: Number of backspaces to send.
    """
    inputs = []
    for _ in range(count):
        inputs.append(_make_key_input(vk=VK_BACK))
        inputs.append(_make_key_input(vk=VK_BACK, flags=KEYEVENTF_KEYUP))
    if inputs:
        _send_inputs(inputs)


def send_unicode_string(text):
    """Send a string as Unicode character events.

    Uses KEYEVENTF_UNICODE so the characters are layout-independent.

    Args:
        text: The string to inject into the active window.
    """
    inputs = []
    for ch in text:
        code = ord(ch)
        inputs.append(_make_key_input(
            scan=code, flags=KEYEVENTF_UNICODE
        ))
        inputs.append(_make_key_input(
            scan=code, flags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
        ))
    if inputs:
        _send_inputs(inputs)


user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.GetKeyboardLayout.restype = wintypes.HKL
user32.GetKeyboardLayout.argtypes = [wintypes.DWORD]
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.PostMessageW.restype = wintypes.BOOL

def toggle_layout(target_layout):
    """Toggle the OS keyboard layout for the foreground window and wait.

    Args:
        target_layout: 'en' or 'he'.
    """
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return

    hkl = HKL_EN_US if target_layout == 'en' else HKL_HE
    user32.PostMessageW(hwnd, WM_INPUTLANGCHANGEREQUEST, 0, hkl)
    
    # Wait for the layout to actually change to prevent race conditions
    thread_id = user32.GetWindowThreadProcessId(hwnd, None)
    expected_lang_id = 0x0409 if target_layout == 'en' else 0x040D
    
    for _ in range(10):  # Wait up to 100ms
        current_hkl = user32.GetKeyboardLayout(thread_id)
        lang_id = current_hkl & 0xFFFF
        if lang_id == expected_lang_id:
            break
        time.sleep(0.01)


def get_current_layout():
    """Detect the currently active keyboard layout."""
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return 'unknown'

    thread_id = user32.GetWindowThreadProcessId(hwnd, None)
    hkl = user32.GetKeyboardLayout(thread_id)
    lang_id = hkl & 0xFFFF

    if lang_id == 0x0409:
        return 'en'
    elif lang_id == 0x040D:
        return 'he'
    return 'unknown'


def execute_switch(buffer_active, buffer_shadow,
                   pending_queue, set_correcting, target_layout):
    """Execute the full concurrency-safe layout correction sequence.

    Steps:
      1. Enter lock state (is_correcting = True).
      2. Erase the incorrect word (backspaces).
      3. Toggle the OS keyboard layout.
      4. Inject the corrected text.
      5. Flush any keys typed during correction.
      6. Release lock state.

    Args:
        buffer_active: The incorrectly typed string to erase.
        buffer_shadow: The correct string to inject.
        pending_queue: collections.deque of (vk_code, shifted) tuples
                       from keys pressed during correction.
        set_correcting: Callable(bool) to set/clear the lock flag.
        target_layout: 'en' or 'he' — the layout to switch TO.
    """
    set_correcting(True)

    try:
        erase_len = len(buffer_active)
        send_backspaces(erase_len)
        time.sleep(0.005)

        toggle_layout(target_layout)

        send_unicode_string(buffer_shadow)
        time.sleep(0.005)

        from core.keymap import vk_to_char

        while pending_queue:
            vk_code, shifted = pending_queue.popleft()
            ch = vk_to_char(vk_code, shifted, layout=target_layout)
            if ch:
                send_unicode_string(ch)
    finally:
        set_correcting(False)
