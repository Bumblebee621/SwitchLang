"""
hooks.py — Low-level keyboard hook (WH_KEYBOARD_LL) and mouse listener.

Intercepts keystrokes, maintains dual-layout buffers, triggers the
evaluation engine, and blocks physical input during correction phases.
"""

import ctypes
import ctypes.wintypes as wintypes
import collections
import logging
import threading
import time

from pynput import mouse as pynput_mouse

from core.keymap import get_both_chars
from core.switcher import execute_switch, get_current_layout

logger = logging.getLogger('switchlang.hooks')

# Dedicated user32 with use_last_error=True (separate from
# ctypes.windll.user32 which may be modified by PyQt6/pynput)
_user32 = ctypes.WinDLL('user32', use_last_error=True)

# Also keep a reference for non-hook calls
user32 = ctypes.windll.user32
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetKeyboardLayout.restype = wintypes.HKL

# Hook constants
WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
HC_ACTION = 0

# Flag to detect injected keys (bit 4 in KBDLLHOOKSTRUCT.flags)
LLKHF_INJECTED = 0x00000010

# Modifier VK codes
VK_SHIFT = 0x10
VK_LSHIFT = 0xA0
VK_RSHIFT = 0xA1
VK_CONTROL = 0x11
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
VK_MENU = 0x12   # Alt
VK_LMENU = 0xA4
VK_RMENU = 0xA5
VK_RETURN = 0x0D
VK_SPACE = 0x20
VK_BACK = 0x08
VK_TAB = 0x09
VK_CAPITAL = 0x14

MODIFIER_VKS = {
    VK_SHIFT, VK_LSHIFT, VK_RSHIFT,
    VK_CONTROL, VK_LCONTROL, VK_RCONTROL,
    VK_MENU, VK_LMENU, VK_RMENU,
}

DELIMITER_VKS = {VK_SPACE, VK_RETURN, VK_TAB}

# Maps delimiter VK codes to their actual characters for re-injection
DELIMITER_CHARS = {VK_SPACE: ' ', VK_RETURN: '\n', VK_TAB: '\t'}

# Stores a completed word and its metadata for retroactive correction
_WordEntry = collections.namedtuple(
    '_WordEntry', ['active', 'shadow', 'delimiter', 'is_ambiguous']
)


# Low-level hook callback type (WINFUNCTYPE = stdcall convention)
HOOKPROC = ctypes.WINFUNCTYPE(
    ctypes.c_long,
    ctypes.c_int,
    wintypes.WPARAM,
    wintypes.LPARAM
)

# Set argtypes/restype on our DEDICATED instance
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


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ('vkCode', wintypes.DWORD),
        ('scanCode', wintypes.DWORD),
        ('flags', wintypes.DWORD),
        ('time', wintypes.DWORD),
        ('dwExtraInfo', ctypes.POINTER(ctypes.c_ulong)),
    ]


class GUITHREADINFO(ctypes.Structure):
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

_user32.GetGUIThreadInfo.argtypes = [wintypes.DWORD, ctypes.POINTER(GUITHREADINFO)]
_user32.GetGUIThreadInfo.restype = wintypes.BOOL
_user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
_user32.GetWindowLongW.restype = wintypes.LONG

def is_password_field_active():
    """Check if the currently focused control has the ES_PASSWORD style."""
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


class HookManager:
    """Manages the global keyboard hook and mouse listener."""

    def __init__(self, engine, sensitivity, blacklist, config):
        """Initialize the hook manager.

        Args:
            engine: EvaluationEngine instance.
            sensitivity: SensitivityManager instance.
            blacklist: BlacklistManager instance.
            config: Dict of configuration values.
        """
        self.engine = engine
        self.sensitivity = sensitivity
        self.blacklist = blacklist
        self.config = config

        self.enabled = config.get('enabled', True)
        self.idle_timeout = config.get('idle_timeout_seconds', 15.0)

        self.buffer_active = ''
        self.buffer_shadow = ''

        # Lookback queue: stores _WordEntry items for completed words in the
        # current CRE session. Cleared on every CRE and after every switch.
        self.history_deque = collections.deque()

        self.is_correcting = False
        self.pending_queue = collections.deque(maxlen=100)

        self._hook_id = None
        self._hook_proc = None
        self._hook_thread = None
        self._mouse_listener = None
        self._running = False

        self._shift_pressed = False
        self._ctrl_pressed = False
        self._on_switch_callback = None

        # Cached values updated by polling thread, NOT in the hook
        self._cached_layout = 'en'
        self._cached_blacklisted = False
        self._last_caps_lock = False

    def set_enabled(self, enabled):
        """Enable or disable the engine."""
        self.enabled = enabled
        logger.info('Engine %s', 'enabled' if enabled else 'disabled')

    def set_on_switch_callback(self, callback):
        """Set a callback for when a switch occurs."""
        self._on_switch_callback = callback

    def _set_correcting(self, val):
        """Set the correction lock flag."""
        self.is_correcting = val

    def _clear_buffers(self):
        """Clear both layout buffers (word-level only)."""
        self.buffer_active = ''
        self.buffer_shadow = ''

    def _clear_history(self):
        """Clear the CRE-scoped word history deque.

        Called on every Context Resumption Event and after every switch
        to prevent retroactive corrections from crossing context boundaries.
        """
        if self.history_deque:
            logger.debug('Clearing history deque (%d entries)', len(self.history_deque))
        self.history_deque.clear()

    def _build_correction_block(self):
        """Collect the contiguous ambiguous words from the tail of the history.

        Walks backward from the most recent word, gathering all entries where
        is_ambiguous=True. Stops at the first non-ambiguous word.

        Returns:
            List of _WordEntry in chronological order (oldest first).
        """
        block = []
        for entry in reversed(self.history_deque):
            if entry.is_ambiguous:
                block.insert(0, entry)
            else:
                break
        if block:
            logger.debug(
                'Correction block: %d ambiguous words: %s',
                len(block), [e.active for e in block]
            )
        return block

    def _handle_keypress(self, vk_code):
        """Process a single keypress through the evaluation pipeline.

        IMPORTANT: This runs inside the LL hook callback. It must be
        fast — no Windows API calls here. We use cached values for
        layout and blacklist status.

        Args:
            vk_code: The virtual key code of the pressed key.

        Returns:
            True to block the key from the OS, False to pass through.
        """
        caps_lock = _user32.GetKeyState(VK_CAPITAL) & 1 == 1

        if self.is_correcting:
            if vk_code in DELIMITER_VKS:
                self.pending_queue.append((vk_code, self._shift_pressed, caps_lock))
            else:
                en_ch, he_ch = get_both_chars(vk_code, self._shift_pressed, caps_lock)
                if en_ch is not None:
                    self.pending_queue.append((vk_code, self._shift_pressed, caps_lock))
            return True

        if not self.enabled:
            return False

        if self._cached_blacklisted:
            return False

        # Skip if Ctrl is held (user is doing Ctrl+C etc.)
        if self._ctrl_pressed:
            return False

        self.sensitivity.record_keystroke()

        if vk_code == VK_BACK:
            if self.buffer_active:
                self.buffer_active = self.buffer_active[:-1]
                self.buffer_shadow = self.buffer_shadow[:-1]
            return False

        if vk_code in DELIMITER_VKS:
            delimiter_char = DELIMITER_CHARS.get(vk_code, ' ')

            if self.buffer_active:
                current = self._cached_layout
                should_switch, diff, is_ambiguous = self.engine.evaluate(
                    self.buffer_active,
                    self.buffer_shadow,
                    self.sensitivity.delta,
                    current_layout=current,
                    on_delimiter=True
                )
                
                logger.debug(
                    'EVAL: "%s" (%s) -> "%s" | diff=%+.2f vs delta=%.2f | switch=%s | ambiguous=%s',
                    self.buffer_active, current, self.buffer_shadow, diff,
                    self.sensitivity.delta, should_switch, is_ambiguous
                )

                needs_caps_fix = current == 'he' and caps_lock and not should_switch

                if should_switch or needs_caps_fix:
                    if self._trigger_switch(delimiter_char=delimiter_char, force_target=current if needs_caps_fix else None):
                        return True

                # Word is not a switch trigger (or switch was aborted) — record it in history.
                self.history_deque.append(_WordEntry(
                    active=self.buffer_active,
                    shadow=self.buffer_shadow,
                    delimiter=delimiter_char,
                    is_ambiguous=is_ambiguous,
                ))

                self.sensitivity.on_word_complete()
            self._clear_buffers()
            return False

        en_char, he_char = get_both_chars(vk_code, self._shift_pressed, caps_lock)
        if en_char is None:
            return False

        current = self._cached_layout
        if current == 'en':
            self.buffer_active += en_char
            self.buffer_shadow += he_char
        elif current == 'he':
            self.buffer_active += he_char
            self.buffer_shadow += en_char
        else:
            self.buffer_active += en_char
            self.buffer_shadow += he_char

        if len(self.buffer_active) >= 3:
            should_switch, diff, is_ambig = self.engine.evaluate(
                self.buffer_active,
                self.buffer_shadow,
                self.sensitivity.delta,
                current_layout=current
            )
            logger.debug(
                'Eval: diff=%.2f delta=%.2f switch=%s',
                diff, self.sensitivity.delta, should_switch
            )
            
            needs_caps_fix = current == 'he' and caps_lock and not should_switch

            if should_switch or needs_caps_fix:
                if self._trigger_switch(force_target=current if needs_caps_fix else None):
                    return True

        return False

    def _trigger_switch(self, delimiter_char=None, force_target=None):
        """Initiate the layout correction sequence.

        Args:
            delimiter_char: The delimiter character that triggered the switch
                (space, newline, tab), or None for mid-word triggers. Used to
                calculate the exact number of backspaces needed.
            force_target: The layout to switch to. If None, it will be the opposite
                of the current layout.

        Returns:
            True if a switch was initiated, False if it was aborted.
        """
        current = self._cached_layout
        target = force_target if force_target is not None else ('he' if current == 'en' else 'en')
        fix_caps = False

        # Build the retroactive correction block BEFORE clearing history.
        correction_block = self._build_correction_block()

        # Handle Caps Lock quirk: typing in Hebrew with Caps Lock ON produces English chars.
        # Engine will evaluate the intended Hebrew correctly because keymap is reverted.
        # s_active is Hebrew (intended), s_shadow is English (what's on screen).
        caps_lock = _user32.GetKeyState(VK_CAPITAL) & 1 == 1
        if current == 'he' and caps_lock:
            # Re-evaluate the logic: if engine says target is English, it means the 
            # user's intent is English (like typing "HELLO"). Per user instructions, 
            # we do NOTHING in this case.
            if target == 'en':
                logger.debug('English intent detected in Hebrew layout with Caps Lock ON - ignoring switch')
                return False

            # If target is still Hebrew, it means the user intended Hebrew (like "לא")
            # but Caps Lock was forcing English. We stay in Hebrew but fix Caps.
            target = 'he'
            fix_caps = True

        logger.info(
            'SWITCHING: "%s" -> "%s" (layout %s -> %s, fix_caps=%s) lookback=%d words',
            self.buffer_active, self.buffer_shadow, current, target,
            fix_caps, len(correction_block)
        )

        buf_active = self.buffer_active
        buf_shadow = self.buffer_shadow
        self._clear_buffers()
        self._clear_history()  # context ends here

        if fix_caps:
            # When fixing Caps Lock without changing layout, the intended text is 
            # already in buf_active, and the incorrectly outputted text on screen 
            # corresponds to buf_shadow. By swapping them, the switcher will erase 
            # the correct number of characters (from shadow) and inject the intended text.
            buf_active, buf_shadow = buf_shadow, buf_active
            correction_block = [
                _WordEntry(active=e.shadow, shadow=e.active, delimiter=e.delimiter, is_ambiguous=e.is_ambiguous)
                for e in correction_block
            ]

        # Synchronously lock OS passthrough before thread spins up
        self._set_correcting(True)

        switch_thread = threading.Thread(
            target=self._do_switch,
            args=(buf_active, buf_shadow, target, correction_block, delimiter_char, fix_caps),
            daemon=True,
            name='SwitchThread'
        )
        switch_thread.start()
        return True

    def _do_switch(self, buf_active, buf_shadow, target,
                   correction_block=None, trigger_delimiter=None,
                   fix_caps=False):
        """Run the switch on a separate thread to avoid blocking
        the hook callback."""
        execute_switch(
            buf_active,
            buf_shadow,
            self.pending_queue,
            self._set_correcting,
            target,
            correction_block=correction_block,
            trigger_delimiter=trigger_delimiter,
            fix_caps=fix_caps,
        )

        self._cached_layout = target
        self.sensitivity.reset(reason='layout_switch')

        if self._on_switch_callback:
            self._on_switch_callback()

    def _kb_hook_callback(self, n_code, w_param, l_param):
        """The WH_KEYBOARD_LL callback function."""
        try:
            if n_code == HC_ACTION:
                kb = ctypes.cast(
                    l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)
                ).contents

                vk = kb.vkCode

                # Skip injected/synthetic keys (our own SendInput)
                if kb.flags & LLKHF_INJECTED:
                    return _user32.CallNextHookEx(
                        self._hook_id, n_code, w_param, l_param
                    )

                if w_param in (WM_KEYDOWN, WM_SYSKEYDOWN):
                    # Track modifier states
                    if vk in (VK_SHIFT, VK_LSHIFT, VK_RSHIFT):
                        self._shift_pressed = True
                    elif vk in (VK_CONTROL, VK_LCONTROL, VK_RCONTROL):
                        self._ctrl_pressed = True
                    elif vk in MODIFIER_VKS:
                        pass
                    else:
                        block = self._handle_keypress(vk)
                        if block:
                            return 1
                else:
                    # Key up
                    if vk in (VK_SHIFT, VK_LSHIFT, VK_RSHIFT):
                        self._shift_pressed = False
                    elif vk in (VK_CONTROL, VK_LCONTROL, VK_RCONTROL):
                        self._ctrl_pressed = False
        except Exception:
            logger.exception('Error in keyboard hook callback')

        return _user32.CallNextHookEx(
            self._hook_id, n_code, w_param, l_param
        )

    def _hook_thread_func(self):
        """Thread function that installs the keyboard hook and
        runs a Windows message pump."""
        self._hook_proc = HOOKPROC(self._kb_hook_callback)

        # WH_KEYBOARD_LL needs a loaded DLL as hMod.
        # python.exe didn't work previously due to 64-bit pointer truncation;
        # setting the proper restype/argtypes fixes it.
        kernel32 = ctypes.windll.kernel32
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        h_mod = kernel32.GetModuleHandleW(None)

        self._hook_id = _user32.SetWindowsHookExW(
            WH_KEYBOARD_LL,
            self._hook_proc,
            h_mod,
            0
        )

        if not self._hook_id:
            err = ctypes.get_last_error()
            logger.error(
                'Failed to install keyboard hook, '
                'error=%d', err
            )
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

    def _on_mouse_click(self, x, y, button, pressed):
        """Mouse click callback — triggers CRE on left/right click.

        Per the Game Plan spec, both left and right mouse buttons indicate
        a caret jump. Middle-click (paste) is excluded as it does not move
        the caret.
        """
        if pressed and button != pynput_mouse.Button.middle:
            logger.debug('--- Mouse Click: Context Resumption Event ---')
            self.sensitivity.reset(reason='mouse_click')
            self._clear_buffers()
            self._clear_history()

    def _poll_foreground_window(self):
        """Periodically update cached layout and blacklist status.

        Runs every 100ms on its own thread so the hook callback
        never needs to call Windows APIs directly.
        """
        while self._running:
            try:
                # Detection of layout, window, or Caps Lock changes as Context Resumption Events
                if not self.is_correcting:
                    new_layout = get_current_layout()
                    if new_layout != 'unknown':
                        if new_layout != self._cached_layout:
                            logger.debug(
                                'Manual layout change detected (%s -> %s) — triggering CRE',
                                self._cached_layout, new_layout
                            )
                            self.sensitivity.reset(reason='manual_layout_change')
                            self._clear_buffers()
                            self._clear_history()
                        self._cached_layout = new_layout

                    # Detect manual Caps Lock toggle
                    caps_state = _user32.GetKeyState(VK_CAPITAL) & 1 == 1
                    if caps_state != self._last_caps_lock:
                        logger.debug('Manual Caps Lock toggle detected — triggering CRE')
                        self.sensitivity.reset(reason='manual_caps_toggle')
                        self._clear_buffers()
                        self._clear_history()
                        self._last_caps_lock = caps_state

                is_blacklisted = self.blacklist.is_blacklisted()
                if not is_blacklisted:
                    try:
                        is_blacklisted = is_password_field_active()
                    except Exception as e:
                        logger.debug('Error checking password field: %s', e)
                self._cached_blacklisted = is_blacklisted

                hwnd = user32.GetForegroundWindow()
                if self.sensitivity.check_window_change(hwnd):
                    logger.debug('--- Window Change: Context Resumption Event ---')
                    self.sensitivity.reset(reason='window_change')
                    self._clear_buffers()
                    self._clear_history()

                # Idle timeout check belongs here, not in the hot hook callback.
                # IMPORTANT: call record_keystroke() after firing so the timer
                # resets — otherwise this triggers every 100ms while idle.
                if self.enabled and self.sensitivity.check_idle_timeout(self.idle_timeout):
                    self.sensitivity.reset(reason='idle_timeout')
                    self.sensitivity.record_keystroke()
                    self._clear_buffers()
                    self._clear_history()
                    logger.debug('Idle timeout — reset sensitivity')

            except Exception:
                logger.exception('Error in foreground poll')

            time.sleep(0.1)

    def start(self):
        """Start all hooks and polling threads."""
        self._running = True

        # Initial layout and caps state detection
        self._cached_layout = get_current_layout()
        self._last_caps_lock = _user32.GetKeyState(VK_CAPITAL) & 1 == 1
        logger.info('Initial layout: %s, Caps Lock: %s', self._cached_layout, self._last_caps_lock)

        self._hook_thread = threading.Thread(
            target=self._hook_thread_func,
            daemon=True,
            name='KeyboardHookThread'
        )
        self._hook_thread.start()

        self._mouse_listener = pynput_mouse.Listener(
            on_click=self._on_mouse_click
        )
        self._mouse_listener.daemon = True
        self._mouse_listener.start()

        self._fg_thread = threading.Thread(
            target=self._poll_foreground_window,
            daemon=True,
            name='ForegroundPollThread'
        )
        self._fg_thread.start()
        logger.info('All hooks and polling threads started')

    def stop(self):
        """Stop all hooks and threads."""
        self._running = False

        if self._hook_id:
            _user32.PostThreadMessageW(
                self._hook_thread.ident,
                0x0012,  # WM_QUIT
                0, 0
            )

        if self._mouse_listener:
            self._mouse_listener.stop()

        if self._hook_thread and self._hook_thread.is_alive():
            self._hook_thread.join(timeout=2.0)

        logger.info('All hooks stopped')
