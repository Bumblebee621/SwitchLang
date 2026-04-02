"""
windows.py — Windows platform backend.

Implements PlatformBackend using Win32 APIs (ctypes).
This is extracted from the original hooks.py, switcher.py, blacklist.py,
startup.py, and main.py — the logic is identical, just reorganised behind
the platform interface.
"""

import ctypes
import ctypes.wintypes as wintypes
import logging
import os
import sys
import time

from pynput import mouse as pynput_mouse

from core.platform.base import PlatformBackend

logger = logging.getLogger('switchlang.platform.windows')

# =============================================================================
# WIN32 CONSTANTS
# =============================================================================

# Hook constants
WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
WM_QUIT = 0x0012
HC_ACTION = 0
LLKHF_INJECTED = 0x00000010

# Virtual Key Constants (used as the normalised keycode space across platforms)
VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_CAPITAL = 0x14
VK_SPACE = 0x20
VK_LSHIFT = 0xA0
VK_RSHIFT = 0xA1
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
VK_LMENU = 0xA4
VK_RMENU = 0xA5

# SendInput constants
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_EXTENDEDKEY = 0x0001
WM_INPUTLANGCHANGEREQUEST = 0x0050

# Process constants
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

# Hardcoded HKL handles (fallback)
HKL_EN_US = 0x04090409
HKL_HE = 0x040D040D

# =============================================================================
# CTYPES STRUCTURES
# =============================================================================


class KBDLLHOOKSTRUCT(ctypes.Structure):
    """Low-level keyboard hook structure from Windows API."""
    _fields_ = [
        ('vkCode', wintypes.DWORD),
        ('scanCode', wintypes.DWORD),
        ('flags', wintypes.DWORD),
        ('time', wintypes.DWORD),
        ('dwExtraInfo', ctypes.POINTER(ctypes.c_ulong)),
    ]


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


# Hook callback type (WINFUNCTYPE = stdcall convention)
HOOKPROC = ctypes.WINFUNCTYPE(
    ctypes.c_long,
    ctypes.c_int,
    wintypes.WPARAM,
    wintypes.LPARAM
)

# =============================================================================
# WIN32 API SETUP
# =============================================================================

# Dedicated user32 with use_last_error=True (separate from
# ctypes.windll.user32 which may be modified by PyQt6/pynput)
_user32 = ctypes.WinDLL('user32', use_last_error=True)

# Also keep a reference for non-hook calls
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi

# Return types
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetKeyboardLayout.restype = wintypes.HKL
user32.GetKeyboardLayout.argtypes = [wintypes.DWORD]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.PostMessageW.restype = wintypes.BOOL
user32.GetKeyboardLayoutList.restype = ctypes.c_int
user32.GetKeyboardLayoutList.argtypes = [ctypes.c_int, ctypes.POINTER(wintypes.HKL)]
user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT
user32.GetGUIThreadInfo.argtypes = [wintypes.DWORD, ctypes.POINTER(GUITHREADINFO)]
user32.GetGUIThreadInfo.restype = wintypes.BOOL
user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowLongW.restype = wintypes.LONG

# Hook function signatures
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

_EXTRA_INFO = ctypes.pointer(ctypes.c_ulong(0))

# Kernel function signatures for process name detection
kernel32.GetModuleHandleW.restype = wintypes.HMODULE
kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# HKL cache for layout toggling
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
        primary_lang = lang_id & 0x03FF

        if lang_id == 0x0409:
            _cached_hkl_en = hkl
        elif lang_id == 0x040D:
            _cached_hkl_he = hkl

        if not _cached_hkl_en and primary_lang == 0x09:
            _cached_hkl_en = hkl
        if not _cached_hkl_he and primary_lang == 0x0D:
            _cached_hkl_he = hkl


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


# =============================================================================
# WINDOWS BACKEND
# =============================================================================


class WindowsBackend(PlatformBackend):
    """Windows platform backend using Win32 APIs."""

    def __init__(self):
        self._hook_id = None
        self._hook_proc = None
        self._hook_thread = None
        self._mouse_listener = None
        self._running = False
        self._on_key_down = None
        self._on_key_up = None
        self._mutex_handle = None

    # ----- Keyboard Hook -----

    def start_keyboard_hook(self, on_key_down, on_key_up):
        import threading

        self._on_key_down = on_key_down
        self._on_key_up = on_key_up
        self._running = True

        self._hook_thread = threading.Thread(
            target=self._hook_thread_func,
            daemon=True,
            name='KeyboardHookThread'
        )
        self._hook_thread.start()

    def stop_keyboard_hook(self):
        self._running = False

        if self._hook_id and self._hook_thread and self._hook_thread.ident:
            _user32.PostThreadMessageW(
                self._hook_thread.ident,
                WM_QUIT,
                0, 0
            )

        if self._hook_thread and self._hook_thread.is_alive():
            self._hook_thread.join(timeout=2.0)

        logger.info('Keyboard hook stopped')

    def _hook_thread_func(self):
        """Thread worker that installs the hook and maintains the Windows message pump."""
        self._hook_proc = HOOKPROC(self._kb_hook_callback)

        h_mod = kernel32.GetModuleHandleW(None)

        self._hook_id = _user32.SetWindowsHookExW(
            WH_KEYBOARD_LL,
            self._hook_proc,
            h_mod,
            0
        )

        if not self._hook_id:
            err = ctypes.get_last_error()
            logger.error('Failed to install keyboard hook, error=%d', err)
            return

        logger.info('Keyboard hook installed (id=%s)', self._hook_id)

        msg = wintypes.MSG()
        while self._running:
            result = _user32.GetMessageW(
                ctypes.byref(msg), None, 0, 0
            )
            if result <= 0:
                break
            _user32.TranslateMessage(ctypes.byref(msg))
            _user32.DispatchMessageW(ctypes.byref(msg))

        _user32.UnhookWindowsHookEx(self._hook_id)
        self._hook_id = None
        logger.info('Keyboard hook removed')

    def _kb_hook_callback(self, n_code, w_param, l_param):
        """The core WH_KEYBOARD_LL callback function."""
        try:
            if n_code == HC_ACTION:
                kb = ctypes.cast(
                    l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)
                ).contents

                vk = kb.vkCode

                # Skip injected keys to avoid infinite loops
                if kb.flags & LLKHF_INJECTED:
                    return _user32.CallNextHookEx(
                        self._hook_id, n_code, w_param, l_param
                    )

                if w_param in (WM_KEYDOWN, WM_SYSKEYDOWN):
                    if self._on_key_down:
                        shifted = bool(_user32.GetKeyState(VK_SHIFT) & 0x8000)
                        block = self._on_key_down(vk, shifted)
                        if block:
                            return 1
                else:
                    # KEY UP events
                    if self._on_key_up:
                        self._on_key_up(vk)
        except Exception:
            logger.exception('Error in keyboard hook callback')

        return _user32.CallNextHookEx(
            self._hook_id, n_code, w_param, l_param
        )

    # ----- Input Injection -----

    def send_backspaces(self, count):
        inputs = []
        for _ in range(count):
            inputs.append(_make_key_input(vk=VK_BACK))
            inputs.append(_make_key_input(vk=VK_BACK, flags=KEYEVENTF_KEYUP))
        if inputs:
            _send_inputs(inputs)

    def send_unicode_string(self, text):
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

    def replace_text(self, erase_count, text):
        self.send_backspaces(erase_count)
        self.send_unicode_string(text)

    def send_key(self, keycode):
        inputs = [
            _make_key_input(vk=keycode),
            _make_key_input(vk=keycode, flags=KEYEVENTF_KEYUP),
        ]
        _send_inputs(inputs)

    def toggle_caps_lock(self):
        inputs = [
            _make_key_input(vk=VK_CAPITAL),
            _make_key_input(vk=VK_CAPITAL, flags=KEYEVENTF_KEYUP)
        ]
        _send_inputs(inputs)

    # ----- Layout Management -----

    def get_current_layout(self):
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return 'unknown'

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

    def toggle_layout(self, target):
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return

        _resolve_hkls()
        hkl = _cached_hkl_en if target == 'en' else _cached_hkl_he

        if not hkl:
            hkl = HKL_EN_US if target == 'en' else HKL_HE

        user32.PostMessageW(hwnd, WM_INPUTLANGCHANGEREQUEST, 0, hkl)

        gui_info = GUITHREADINFO(cbSize=ctypes.sizeof(GUITHREADINFO))
        thread_id = 0
        if user32.GetGUIThreadInfo(0, ctypes.byref(gui_info)) and gui_info.hwndFocus:
            thread_id = user32.GetWindowThreadProcessId(gui_info.hwndFocus, None)

        if not thread_id:
            thread_id = user32.GetWindowThreadProcessId(hwnd, None)

        expected_primary = 0x09 if target == 'en' else 0x0D

        for _ in range(10):
            current_hkl = user32.GetKeyboardLayout(thread_id)
            lang_id = current_hkl & 0xFFFF
            primary_lang = lang_id & 0x03FF
            if primary_lang == expected_primary:
                break
            time.sleep(0.01)

    # ----- System Queries -----

    def get_foreground_process(self):
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

    def is_password_field_active(self):
        gui_info = GUITHREADINFO(cbSize=ctypes.sizeof(GUITHREADINFO))
        if user32.GetGUIThreadInfo(0, ctypes.byref(gui_info)):
            if gui_info.hwndFocus:
                GWL_STYLE = -16
                ES_PASSWORD = 0x0020
                style = user32.GetWindowLongW(gui_info.hwndFocus, GWL_STYLE)
                if style & ES_PASSWORD:
                    return True
        return False

    def is_caps_lock_on(self):
        return _user32.GetKeyState(VK_CAPITAL) & 1 == 1

    # ----- Mouse Listener -----

    def start_mouse_listener(self, on_click):
        self._mouse_listener = pynput_mouse.Listener(on_click=on_click)
        self._mouse_listener.daemon = True
        self._mouse_listener.start()

    def stop_mouse_listener(self):
        if self._mouse_listener:
            self._mouse_listener.stop()

    # ----- Application Lifecycle -----

    def set_single_instance_lock(self, app_name):
        mutex_name = f"Local\\{app_name}_Mutex_v1"
        ERROR_ALREADY_EXISTS = 183

        self._mutex_handle = kernel32.CreateMutexW(None, False, mutex_name)
        last_error = kernel32.GetLastError()

        if last_error == ERROR_ALREADY_EXISTS:
            return False
        return True

    def release_single_instance_lock(self):
        if self._mutex_handle:
            kernel32.CloseHandle(self._mutex_handle)
            self._mutex_handle = None

    def get_config_dir(self):
        appdata = os.getenv('APPDATA', os.path.expanduser('~'))
        return os.path.join(appdata, 'SwitchLang')

    def is_startup_enabled(self, app_name='SwitchLang'):
        import winreg
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ) as key:
                try:
                    value, _ = winreg.QueryValueEx(key, app_name)
                    current_cmd = self._get_current_app_command()
                    return current_cmd.lower() in value.lower() or value.lower() in current_cmd.lower()
                except FileNotFoundError:
                    return False
        except Exception as e:
            logger.error('Error checking startup registry: %s', e)
            return False

    def set_startup_enabled(self, enabled, app_name='SwitchLang'):
        import winreg
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0,
                                winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE) as key:
                if enabled:
                    cmd = self._get_current_app_command()
                    winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, cmd)
                    logger.info('Enabled startup: %s', cmd)
                else:
                    try:
                        winreg.DeleteValue(key, app_name)
                        logger.info('Disabled startup.')
                    except FileNotFoundError:
                        pass
            return True
        except Exception as e:
            logger.error('Error modifying startup registry: %s', e)
            return False

    def set_app_id(self, app_id):
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception:
            pass

    def _get_current_app_command(self):
        """Returns the command line string to run the current application."""
        if getattr(sys, 'frozen', False):
            return f'"{sys.executable}"'
        else:
            python_exe = sys.executable
            script_path = os.path.abspath(sys.argv[0])
            return f'"{python_exe}" "{script_path}"'

    # ----- Keycode Translation (identity — VK is the normalised space) -----

    def translate_keycode(self, native_keycode):
        return native_keycode

    def get_native_keycode(self, vk_code):
        return vk_code
