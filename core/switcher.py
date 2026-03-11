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

from core.keymap import vk_to_char

logger = logging.getLogger('switchlang.switcher')

user32 = ctypes.windll.user32

# Ensure correct return types for 64-bit compatibility
user32.GetForegroundWindow.restype = wintypes.HWND

# Windows constants
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
VK_BACK = 0x08
VK_CAPITAL = 0x14
WM_INPUTLANGCHANGEREQUEST = 0x0050
KEYEVENTF_EXTENDEDKEY = 0x0001

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

# HARDWAREINPUT is omitted; only KEYBDINPUT is needed here.

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


def toggle_caps_lock():
    """Simulate a Caps Lock key press to toggle its state."""
    inputs = [
        _make_key_input(vk=VK_CAPITAL, flags=KEYEVENTF_EXTENDEDKEY),
        _make_key_input(vk=VK_CAPITAL, flags=KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP)
    ]
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
    """Detect the currently active keyboard layout."""
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return 'unknown'

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
                   pending_queue, set_correcting, target_layout,
                   correction_block=None, trigger_delimiter=None,
                   fix_caps=False):
    """Execute the full concurrency-safe layout correction sequence.

    Steps:
      1. Enter lock state (is_correcting = True).
      2. Erase the incorrect text (current word + any retroactive block).
      3. Toggle the OS keyboard layout (if needed) or Caps Lock.
      4. Inject the full corrected text.
      5. Flush any keys typed during correction.
      6. Release lock state.

    Args:
        buffer_active: The incorrectly typed trigger word to erase.
        buffer_shadow: The shadow translation of the trigger word.
        pending_queue: collections.deque of (vk_code, shifted, caps_lock) tuples
                       from keys pressed during correction.
        set_correcting: Callable(bool) to set/clear the lock flag.
        target_layout: 'en' or 'he' — the layout to switch TO.
        correction_block: Optional list of _WordEntry namedtuples for
                          retroactively corrected ambiguous words that
                          preceded the trigger word.
        trigger_delimiter: The delimiter char (e.g. ' ') that was passed
                           through to the app before the switch thread ran,
                           or None for mid-word triggers.
        fix_caps: If True, toggle Caps Lock off during the correction.
    """
    set_correcting(True)

    try:
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
            if trigger_delimiter:
                parts.append(trigger_delimiter)
            inject_text = ''.join(parts)

            if correction_block:
                logger.info(
                    'Retroactive correction: erasing %d chars, injecting "%s"',
                    erase_len, inject_text
                )
                for entry in correction_block:
                    logger.info('  └─ Replaced history: "%s" -> "%s"', entry.active, entry.shadow)
            else:
                logger.info(
                    'Correction: erasing %d chars, injecting "%s"',
                    erase_len, inject_text
                )
        else:
            erase_len = len(buffer_active)
            if trigger_delimiter is None:
                erase_len -= 1
            
            inject_text = buffer_shadow
            if trigger_delimiter:
                inject_text += trigger_delimiter

        send_backspaces(erase_len)
        time.sleep(0.005)

        if fix_caps:
            toggle_caps_lock()
            time.sleep(0.005)

        current_layout = get_current_layout()
        if target_layout != current_layout:
            toggle_layout(target_layout)

        send_unicode_string(inject_text)
        time.sleep(0.005)
        # Inject pending queue
        text_to_inject = ""
        while pending_queue:
            q_vk, q_shift, q_caps = pending_queue.popleft()

            # Map the raw typed key to a character in the TARGET layout.
            ch = vk_to_char(q_vk, q_shift, layout=target_layout, caps_lock=q_caps)
            if ch:
                text_to_inject += ch
        if text_to_inject:
            send_unicode_string(text_to_inject)
    finally:
        set_correcting(False)
