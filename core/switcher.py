"""
switcher.py — Concurrency-safe keyboard layout correction sequence.

Handles the critical section: erasing wrong text, toggling the OS
keyboard layout, injecting corrected text, and flushing any keys
typed during the correction phase.
"""

import ctypes
import ctypes.wintypes as wintypes
import logging
import time

from core.keymap import CHAR_TO_VK_EN, CHAR_TO_VK_HE

logger = logging.getLogger('switchlang.switcher')

user32 = ctypes.windll.user32

# Ensure correct return types for 64-bit compatibility
user32.GetForegroundWindow.restype = wintypes.HWND

# Windows constants
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
VK_BACK = 0x08
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CAPITAL = 0x14
WM_INPUTLANGCHANGEREQUEST = 0x0050
KEYEVENTF_EXTENDEDKEY = 0x0001
_CORRECTION_STEP_DELAY = 0.005  # seconds between correction sub-steps

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

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ('wVk', wintypes.WORD),
        ('wScan', wintypes.WORD),
        ('dwFlags', wintypes.DWORD),
        ('time', wintypes.DWORD),
        ('dwExtraInfo', ctypes.POINTER(ctypes.c_ulong)),
    ]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ('uMsg', wintypes.DWORD),
        ('wParamL', wintypes.WORD),
        ('wParamH', wintypes.WORD),
    ]

class INPUT(ctypes.Structure):
    class _INPUT_UNION(ctypes.Union):
        _fields_ = [
            ('ki', KEYBDINPUT),
            ('mi', MOUSEINPUT),
            ('hi', HARDWAREINPUT),
        ]

    _fields_ = [
        ('type', wintypes.DWORD),
        ('union', _INPUT_UNION),
    ]

_EXTRA_INFO = ctypes.pointer(ctypes.c_ulong(0))

class GUITHREADINFO(ctypes.Structure):
    """GUI thread information used to detect focused control properties."""
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hwndActive", wintypes.HWND),
        ("hwndFocus", wintypes.HWND),
        ("hwndCapture", wintypes.HWND),
        ("hwndMenuOwner", wintypes.HWND),
        ("hwndMoveSize", wintypes.HWND),
        ("hwndCaret", wintypes.HWND),
        ("rcCaret", wintypes.RECT)
    ]

# Set argtypes and restype for critical user32 functions
user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT
user32.GetGUIThreadInfo.argtypes = [wintypes.DWORD, ctypes.POINTER(GUITHREADINFO)]
user32.GetGUIThreadInfo.restype = wintypes.BOOL


def _make_key_input(vk=0, scan=0, flags=0):
    """Create an INPUT structure for a keyboard event."""
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki.wVk = vk
    inp.union.ki.wScan = scan
    inp.union.ki.dwFlags = flags
    inp.union.ki.time = 0
    inp.union.ki.dwExtraInfo = _EXTRA_INFO
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


def toggle_caps_lock():
    """Simulate a Caps Lock key press to toggle its state."""
    inputs = [
        _make_key_input(vk=VK_CAPITAL),
        _make_key_input(vk=VK_CAPITAL, flags=KEYEVENTF_KEYUP)
    ]
    _send_inputs(inputs)


def send_vk_key(vk):
    """Send a single virtual key press (down + up) as a real VK event.

    Unlike send_unicode_string, this sends the key using its virtual key code,
    so apps that listen for specific VK events (like Discord listening for
    VK_RETURN to send a message) will recognise it.
    """
    inputs = [
        _make_key_input(vk=vk),
        _make_key_input(vk=vk, flags=KEYEVENTF_KEYUP),
    ]
    _send_inputs(inputs)


def send_string_as_keys(text, layout):
    """Send text as real VK key events through the active OS keyboard layout.

    Unlike KEYEVENTF_UNICODE (which relies on VK_PACKET → TranslateMessage),
    this sends real keyboard events through the normal OS input pipeline.
    Modern WinUI/XAML apps (e.g. Windows 11 Notepad) handle these reliably,
    whereas they can garble batched KEYEVENTF_UNICODE events.

    Falls back to KEYEVENTF_UNICODE for characters not in the VK mapping.

    Args:
        text: The string to inject into the active window.
        layout: 'en' or 'he' — the currently active OS layout.
    """
    table = CHAR_TO_VK_EN if layout == 'en' else CHAR_TO_VK_HE

    for ch in text:
        entry = table.get(ch)
        if entry:
            vk, shifted = entry
            inputs = []
            if shifted:
                inputs.append(_make_key_input(vk=VK_SHIFT))
            inputs.append(_make_key_input(vk=vk))
            inputs.append(_make_key_input(vk=vk, flags=KEYEVENTF_KEYUP))
            if shifted:
                inputs.append(_make_key_input(vk=VK_SHIFT, flags=KEYEVENTF_KEYUP))
            _send_inputs(inputs)
        else:
            # Fallback: use Unicode for chars not in VK mapping
            code = ord(ch)
            pair = [
                _make_key_input(scan=code, flags=KEYEVENTF_UNICODE),
                _make_key_input(scan=code, flags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP),
            ]
            _send_inputs(pair)


user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.GetKeyboardLayout.restype = wintypes.HKL
user32.GetKeyboardLayout.argtypes = [wintypes.DWORD]
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.PostMessageW.restype = wintypes.BOOL
user32.GetKeyboardLayoutList.restype = ctypes.c_int
user32.GetKeyboardLayoutList.argtypes = [ctypes.c_int, ctypes.POINTER(wintypes.HKL)]

_cached_hkl_en = None
_cached_hkl_he = None

def _resolve_hkls():
    """Discover the actual HKL handles for English and Hebrew on this system."""
    global _cached_hkl_en, _cached_hkl_he
    if _cached_hkl_en and _cached_hkl_he:
        return

    count = user32.GetKeyboardLayoutList(0, None)
    if count == 0:
        return

    hkl_array = (wintypes.HKL * count)()
    user32.GetKeyboardLayoutList(count, hkl_array)

    for hkl in hkl_array:
        lang_id = ctypes.c_ulong(hkl).value & 0xFFFF
        # Per MAKELANGID (WinNT.h): primary lang = low 10 bits of LANGID
        primary_lang = lang_id & 0x03FF
        
        if lang_id == 0x0409:
            _cached_hkl_en = hkl
        elif lang_id == 0x040D:
            _cached_hkl_he = hkl

        # Fallback: accept any sublang of the target primary language
        if not _cached_hkl_en and primary_lang == 0x09:
            _cached_hkl_en = hkl
        if not _cached_hkl_he and primary_lang == 0x0D:
            _cached_hkl_he = hkl

def toggle_layout(target_layout):
    """Toggle the OS keyboard layout for the foreground window and wait.

    Args:
        target_layout: 'en' or 'he'.
    """
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return

    _resolve_hkls()
    hkl = _cached_hkl_en if target_layout == 'en' else _cached_hkl_he
    
    # If we couldn't resolve the HKL, fallback to the hardcoded constants
    if not hkl:
        hkl = HKL_EN_US if target_layout == 'en' else HKL_HE

    user32.PostMessageW(hwnd, WM_INPUTLANGCHANGEREQUEST, 0, hkl)
    
    # Wait for the layout to actually change to prevent race conditions
    # We check the thread of the focused component if possible, as it's the priority
    gui_info = GUITHREADINFO(cbSize=ctypes.sizeof(GUITHREADINFO))
    thread_id = 0
    if user32.GetGUIThreadInfo(0, ctypes.byref(gui_info)) and gui_info.hwndFocus:
        thread_id = user32.GetWindowThreadProcessId(gui_info.hwndFocus, None)
    
    if not thread_id:
        thread_id = user32.GetWindowThreadProcessId(hwnd, None)

    expected_primary = 0x09 if target_layout == 'en' else 0x0D
    
    for _ in range(10):  # Wait up to 100ms
        current_hkl = user32.GetKeyboardLayout(thread_id)
        lang_id = current_hkl & 0xFFFF
        primary_lang = lang_id & 0x03FF
        if primary_lang == expected_primary:
            break
        time.sleep(0.01)


def get_current_layout():
    """Detect the currently active keyboard layout.
    
    This method prioritizes the thread of the actually focused control (via GetGUIThreadInfo)
    to handle modern apps like Notepad where the foreground window thread might not 
    reflect the active input locale.
    """
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return 'unknown'

    # Try to get the focused window thread for accurate layout in modern apps
    gui_info = GUITHREADINFO(cbSize=ctypes.sizeof(GUITHREADINFO))
    thread_id = 0
    if user32.GetGUIThreadInfo(0, ctypes.byref(gui_info)) and gui_info.hwndFocus:
        thread_id = user32.GetWindowThreadProcessId(gui_info.hwndFocus, None)
    
    if not thread_id:
        thread_id = user32.GetWindowThreadProcessId(hwnd, None)

    hkl = user32.GetKeyboardLayout(thread_id)
    lang_id = hkl & 0xFFFF
    primary_lang = lang_id & 0x03FF

    if primary_lang == 0x09:
        return 'en'
    elif primary_lang == 0x0D:
        return 'he'
    return 'unknown'


def execute_switch(buffer_active, buffer_shadow,
                   target_layout,
                   correction_block=None, trigger_delimiter=None,
                   fix_caps=False):
    """Execute the erase/toggle/inject correction sequence.

    NOTE: The caller (_do_switch) is responsible for the is_correcting lock
    lifecycle and for draining/integrating the pending_queue. This function
    only handles the OS-level correction steps.

    Steps:
      1. Erase the incorrect text (current word + any retroactive block).
      2. Toggle the OS keyboard layout (if needed) or Caps Lock.
      3. Inject the full corrected text.

    Args:
        buffer_active: The incorrectly typed trigger word to erase.
        buffer_shadow: The shadow translation of the trigger word.
        target_layout: 'en' or 'he' — the layout to switch TO.
        correction_block: Optional list of _WordEntry namedtuples for
                          retroactively corrected colliding/ambiguous words that
                          preceded the trigger word.
        trigger_delimiter: The delimiter char (e.g. ' ') that was passed
                           through to the app before the switch thread ran,
                           or None for mid-word triggers.
        fix_caps: If True, toggle Caps Lock off during the correction.
    """
    # Determine if the trigger delimiter needs a real VK event.
    # Apps like Discord/Chrome require a real VK_RETURN to send a message;
    # a Unicode '\n' injected via KEYEVENTF_UNICODE is silently ignored.
    needs_vk_return = (trigger_delimiter == '\n')

    if correction_block:
        # Erase: trigger word (minus trigger char if mid-word) +
        #        each ambiguous word and the delimiter after it.
        # (If trigger_delimiter is set, it was blocked, so it's not in the OS buffer)
        erase_len = len(buffer_active)
        if trigger_delimiter is None:
            erase_len -= 1
        for entry in correction_block:
            erase_len += len(entry.active) + len(entry.delimiter)

        # Inject: ambiguous shadows + their original delimiters +
        #         the trigger word's shadow + the trigger delimiter.
        parts = []
        for entry in correction_block:
            parts.append(entry.shadow)
            parts.append(entry.delimiter)
        parts.append(buffer_shadow)
        if trigger_delimiter and not needs_vk_return:
            parts.append(trigger_delimiter)
        inject_text = ''.join(parts)

        logger.info(
            'Retroactive correction: erasing %d chars, injecting "%s"',
            erase_len, inject_text
        )
        for entry in correction_block:
            logger.info('  └─ Replaced history: "%s" -> "%s"', entry.active, entry.shadow)
    else:
        erase_len = len(buffer_active)
        if trigger_delimiter is None:
            erase_len -= 1
        
        inject_text = buffer_shadow
        if trigger_delimiter and not needs_vk_return:
            inject_text += trigger_delimiter

        logger.info(
            'Correction: erasing %d chars, injecting "%s"',
            erase_len, inject_text
        )

    send_backspaces(erase_len)
    time.sleep(_CORRECTION_STEP_DELAY)

    if fix_caps:
        toggle_caps_lock()
        time.sleep(_CORRECTION_STEP_DELAY)

    current_layout = get_current_layout()
    if target_layout != current_layout:
        toggle_layout(target_layout)

    send_string_as_keys(inject_text, target_layout)
    time.sleep(_CORRECTION_STEP_DELAY)

    # Send Enter as a real VK keypress so apps like Discord recognise it.
    if needs_vk_return:
        send_vk_key(VK_RETURN)
        time.sleep(_CORRECTION_STEP_DELAY)
