"""
os_windows.py — Windows input layer (Win32 via ctypes).

Low-level keyboard/mouse hooks (WH_KEYBOARD_LL / WH_MOUSE_LL), key injection
(SendInput), keyboard layout switching (HKL), foreground-window and password
field detection. Exposes the interface listed in __all__, shared with
os_linux / os_macos and consumed through core.osapi.
"""

import ctypes
import ctypes.wintypes as wintypes
import logging
import os
import threading
import time

from core.keymap import CHAR_TO_VK_EN, CHAR_TO_VK_HE

__all__ = [
    'start_hooks', 'stop_hooks', 'is_caps_lock_on', 'is_password_field_active',
    'get_foreground_window', 'get_foreground_exe', 'get_current_layout', 'toggle_layout',
    'send_backspaces', 'toggle_caps_lock', 'send_vk_key', 'send_string_as_keys',
    'native_key_to_vk',
]

logger = logging.getLogger('switchlang.os_windows')

# =============================================================================
# SECTION 1: WINDOWS API DEFINITIONS (ctypes)
# =============================================================================

# Dedicated user32 with use_last_error=True (separate from
# ctypes.windll.user32 which may be modified by PyQt6)
_user32 = ctypes.WinDLL('user32', use_last_error=True)

# Also keep a reference for non-hook calls
user32 = ctypes.windll.user32
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetKeyboardLayout.restype = wintypes.HKL

kernel32 = ctypes.windll.kernel32
kernel32.GetModuleHandleW.restype = wintypes.HMODULE
kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]

# Hook constants
WH_KEYBOARD_LL = 13
WH_MOUSE_LL = 14
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
WM_QUIT = 0x0012
HC_ACTION = 0

# Mouse event constants
WM_LBUTTONDOWN = 0x0201
WM_RBUTTONDOWN = 0x0204
WM_MBUTTONDOWN = 0x0207
WM_XBUTTONDOWN = 0x020B

# Flag to detect injected keys (bit 4 in KBDLLHOOKSTRUCT.flags)
# We use this to avoid infinite loops when we re-inject corrected characters.
LLKHF_INJECTED = 0x00000010

# Low-level hook callback type (WINFUNCTYPE = stdcall convention)
HOOKPROC = ctypes.WINFUNCTYPE(
    ctypes.c_long,
    ctypes.c_int,
    wintypes.WPARAM,
    wintypes.LPARAM
)

# Set argtypes/restype for the Windows Hook functions
_user32.SetWindowsHookExW.argtypes = [
    ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD
]
_user32.SetWindowsHookExW.restype = wintypes.HHOOK

_user32.CallNextHookEx.argtypes = [
    wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM
]
_user32.CallNextHookEx.restype = ctypes.c_long

_user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
_user32.UnhookWindowsHookEx.restype = wintypes.BOOL

_user32.GetMessageW.argtypes = [
    ctypes.POINTER(wintypes.MSG), wintypes.HWND,
    wintypes.UINT, wintypes.UINT
]
_user32.GetMessageW.restype = wintypes.BOOL

_user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
_user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]

_user32.PostThreadMessageW.argtypes = [
    wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
]
_user32.PostThreadMessageW.restype = wintypes.BOOL


def _post_quit(thread_id, max_retries=10, delay=0.05):
    """Post WM_QUIT to a thread message queue, retrying if the queue isn't created yet."""
    for _ in range(max_retries):
        if _user32.PostThreadMessageW(thread_id, WM_QUIT, 0, 0):
            return True
        time.sleep(delay)
    return False


class KBDLLHOOKSTRUCT(ctypes.Structure):
    """Low-level keyboard hook structure from Windows API."""
    _fields_ = [
        ('vkCode', wintypes.DWORD),
        ('scanCode', wintypes.DWORD),
        ('flags', wintypes.DWORD),
        ('time', wintypes.DWORD),
        ('dwExtraInfo', ctypes.POINTER(ctypes.c_ulong)),
    ]


# Windows constants
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
VK_BACK = 0x08
VK_SHIFT = 0x10
VK_CAPITAL = 0x14
WM_INPUTLANGCHANGEREQUEST = 0x0050
KEYEVENTF_EXTENDEDKEY = 0x0001

# Hebrew and English HKL handles
HKL_EN_US = 0x04090409
HKL_HE = 0x040D040D

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


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

_user32.GetGUIThreadInfo.argtypes = [wintypes.DWORD, ctypes.POINTER(GUITHREADINFO)]
_user32.GetGUIThreadInfo.restype = wintypes.BOOL
_user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
_user32.GetWindowLongW.restype = wintypes.LONG


# =============================================================================
# SECTION 2: KEY INJECTION
# =============================================================================

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


def send_vk_key(vk, shift=False):
    """Send a single virtual key press (down + up) as a real VK event with optional modifiers."""
    inputs = []
    if shift:
        inputs.append(_make_key_input(vk=VK_SHIFT))
    inputs.append(_make_key_input(vk=vk))
    inputs.append(_make_key_input(vk=vk, flags=KEYEVENTF_KEYUP))
    if shift:
        inputs.append(_make_key_input(vk=VK_SHIFT, flags=KEYEVENTF_KEYUP))
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


# =============================================================================
# SECTION 3: KEYBOARD LAYOUT (HKL)
# =============================================================================

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

def _get_active_thread_id(hwnd):
    """Retrieve thread ID of the focused control or fallback to hwnd."""
    gui_info = GUITHREADINFO(cbSize=ctypes.sizeof(GUITHREADINFO))
    if user32.GetGUIThreadInfo(0, ctypes.byref(gui_info)) and gui_info.hwndFocus:
        tid = user32.GetWindowThreadProcessId(gui_info.hwndFocus, None)
        if tid:
            return tid
    return user32.GetWindowThreadProcessId(hwnd, None) if hwnd else 0


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
    thread_id = _get_active_thread_id(hwnd)

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

    thread_id = _get_active_thread_id(hwnd)

    hkl = user32.GetKeyboardLayout(thread_id)
    lang_id = hkl & 0xFFFF
    primary_lang = lang_id & 0x03FF

    if primary_lang == 0x09:
        return 'en'
    elif primary_lang == 0x0D:
        return 'he'
    return 'unknown'


# =============================================================================
# SECTION 4: UI AUTOMATION HELPERS
# =============================================================================

def is_caps_lock_on():
    """Helper to check if Caps Lock is currently activated."""
    return _user32.GetKeyState(VK_CAPITAL) & 1 == 1


def is_password_field_active():
    """Safety check: detects if the user is currently typing in a password field.

    Returns:
        True if the currently focused window control has the ES_PASSWORD style.
    """
    gui_info = GUITHREADINFO(cbSize=ctypes.sizeof(GUITHREADINFO))
    # 0 for the foreground thread
    if _user32.GetGUIThreadInfo(0, ctypes.byref(gui_info)):
        if gui_info.hwndFocus:
            GWL_STYLE = -16
            ES_PASSWORD = 0x0020
            style = _user32.GetWindowLongW(gui_info.hwndFocus, GWL_STYLE)
            if style & ES_PASSWORD:
                return True
    return False


def get_foreground_window():
    """Handle (HWND) of the foreground window; hashable, 0/None when there is none."""
    return user32.GetForegroundWindow()


def get_foreground_exe():
    """Get the executable name of the currently focused window.

    Returns:
        The .exe filename (e.g. 'notepad.exe'), or '' on failure.
    """
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return ''

    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

    if pid.value == 0:
        return ''

    h_process = kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value
    )
    if not h_process:
        return ''

    try:
        buf = ctypes.create_unicode_buffer(512)
        size = wintypes.DWORD(512)
        success = kernel32.QueryFullProcessImageNameW(
            h_process, 0, buf, ctypes.byref(size)
        )
        if success:
            full_path = buf.value
            return os.path.basename(full_path).lower()
        return ''
    finally:
        kernel32.CloseHandle(h_process)


def native_key_to_vk(event):
    """Windows virtual-key code of a Qt key event."""
    return event.nativeVirtualKey()


# =============================================================================
# SECTION 5: LOW-LEVEL HOOK ENGINE (Message Pumps & Threads)
# =============================================================================

_on_key = None
_on_click = None
_running = False

_hook_id = None
_hook_proc = None
_hook_thread = None
_mouse_hook_id = None
_mouse_hook_proc = None
_mouse_thread = None


def _kb_hook_callback(n_code, w_param, l_param):
    """The core WH_KEYBOARD_LL callback function.

    This is the most time-sensitive block of code in the entire application.
    Failure to call NextHookEx quickly enough will lag the whole OS.
    """
    try:
        if n_code == HC_ACTION:
            kb = ctypes.cast(
                l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)
            ).contents

            # Important: Do not re-process keys we injected ourselves.
            if kb.flags & LLKHF_INJECTED:
                return _user32.CallNextHookEx(
                    _hook_id, n_code, w_param, l_param
                )

            if _on_key(kb.vkCode, w_param in (WM_KEYDOWN, WM_SYSKEYDOWN)):
                return 1  # Stop the key from reaching the original app.
    except Exception:
        logger.exception('Error in keyboard hook callback')

    # Always pass control to the next hook in the chain unless explicitly blocked.
    return _user32.CallNextHookEx(
        _hook_id, n_code, w_param, l_param
    )


def _hook_thread_func():
    """Thread worker that installs the hook and maintains the Windows message pump."""
    global _hook_proc, _hook_id
    _hook_proc = HOOKPROC(_kb_hook_callback)
    h_mod = kernel32.GetModuleHandleW(None)

    _hook_id = _user32.SetWindowsHookExW(
        WH_KEYBOARD_LL,
        _hook_proc,
        h_mod,
        0
    )

    if not _hook_id:
        err = ctypes.get_last_error()
        logger.error('Failed to install keyboard hook, error=%d', err)
        return

    logger.info('Keyboard hook installed (id=%s)', _hook_id)

    # The message pump is required for low-level hooks to receive events.
    msg = wintypes.MSG()
    while _running:
        result = _user32.GetMessageW(
            ctypes.byref(msg), None, 0, 0
        )
        if result <= 0:
            break
        _user32.TranslateMessage(ctypes.byref(msg))
        _user32.DispatchMessageW(ctypes.byref(msg))

    _user32.UnhookWindowsHookEx(_hook_id)
    _hook_id = None
    logger.info('Keyboard hook removed')


def _mouse_hook_callback(n_code, w_param, l_param):
    """Low-level mouse hook callback — reports mouse clicks."""
    try:
        if n_code == HC_ACTION:
            # Trigger on Left, Right, or X-button down (exclude middle click, matching previous behavior)
            if w_param in (WM_LBUTTONDOWN, WM_RBUTTONDOWN, WM_XBUTTONDOWN):
                _on_click()
    except Exception:
        logger.exception('Error in mouse hook callback')

    return _user32.CallNextHookEx(
        _mouse_hook_id, n_code, w_param, l_param
    )


def _mouse_thread_func():
    """Thread worker that installs the mouse hook and maintains a Windows message pump."""
    global _mouse_hook_proc, _mouse_hook_id
    _mouse_hook_proc = HOOKPROC(_mouse_hook_callback)
    h_mod = kernel32.GetModuleHandleW(None)

    _mouse_hook_id = _user32.SetWindowsHookExW(
        WH_MOUSE_LL,
        _mouse_hook_proc,
        h_mod,
        0
    )

    if not _mouse_hook_id:
        err = ctypes.get_last_error()
        logger.error('Failed to install mouse hook, error=%d', err)
        return

    logger.info('Mouse hook installed (id=%s)', _mouse_hook_id)

    msg = wintypes.MSG()
    while _running:
        result = _user32.GetMessageW(
            ctypes.byref(msg), None, 0, 0
        )
        if result <= 0:
            break
        _user32.TranslateMessage(ctypes.byref(msg))
        _user32.DispatchMessageW(ctypes.byref(msg))

    _user32.UnhookWindowsHookEx(_mouse_hook_id)
    _mouse_hook_id = None
    logger.info('Mouse hook removed')


def start_hooks(on_key, on_click):
    """Install the keyboard and mouse hooks, each on its own message-pump thread.

    Args:
        on_key: on_key(vk, is_down) -> bool; return True to block the key.
        on_click: on_click() with no arguments; called on mouse button down.
    """
    global _on_key, _on_click, _running, _hook_thread, _mouse_thread
    _on_key = on_key
    _on_click = on_click
    _running = True

    # Thread 1: The hot keyboard hook
    _hook_thread = threading.Thread(
        target=_hook_thread_func,
        daemon=True,
        name='KeyboardHookThread'
    )
    _hook_thread.start()

    # Thread 2: Mouse listener (native Win32 WH_MOUSE_LL)
    _mouse_thread = threading.Thread(
        target=_mouse_thread_func,
        daemon=True,
        name='MouseHookThread'
    )
    _mouse_thread.start()


def stop_hooks():
    """Safely dismantle both hooks and join their threads."""
    global _running
    _running = False

    if _hook_thread and _hook_thread.ident:
        _post_quit(_hook_thread.ident)

    if _mouse_thread and _mouse_thread.ident:
        _post_quit(_mouse_thread.ident)

    if _hook_thread and _hook_thread.is_alive():
        _hook_thread.join(timeout=2.0)

    if _mouse_thread and _mouse_thread.is_alive():
        _mouse_thread.join(timeout=2.0)
