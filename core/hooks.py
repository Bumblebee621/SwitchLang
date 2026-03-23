"""
hooks.py — Low-level keyboard hook (WH_KEYBOARD_LL) and mouse listener.

This module is the "sensory system" of SwitchLang. It uses Windows Hooks to
intercept every physical keystroke, processes them through an N-gram evaluation
pipeline, and triggers layout corrections when gibberish is detected.

KEY CONCEPTS:
- Low-Level Hook: Intercepts keys before the OS or other apps see them.
- Shadow Buffer: Keeps track of what the keys WOULD be in the other layout.
- Context Resumption Event (CRE): Resets buffers and sensitivity when the
  user switches windows, clicks the mouse, or stays idle, signaling a "fresh start".
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

# =============================================================================
# SECTION 1: WINDOWS API DEFINITIONS (ctypes)
# =============================================================================

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
WM_QUIT = 0x0012
HC_ACTION = 0

# Flag to detect injected keys (bit 4 in KBDLLHOOKSTRUCT.flags)
# We use this to avoid infinite loops when we re-inject corrected characters.
LLKHF_INJECTED = 0x00000010

# Virtual Key (VK) Constants
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
VK_A = 0x41   # Used for Ctrl+A CRE

MODIFIER_VKS = {
    VK_SHIFT, VK_LSHIFT, VK_RSHIFT,
    VK_CONTROL, VK_LCONTROL, VK_RCONTROL,
    VK_MENU, VK_LMENU, VK_RMENU,
}

# Delimiters trigger word-level evaluation
DELIMITER_VKS = {VK_SPACE, VK_RETURN}
DELIMITER_CHARS = {VK_SPACE: ' ', VK_RETURN: '\n'}

# Stores a completed word and its metadata for retroactive correction (Lookback)
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

_user32.GetGUIThreadInfo.argtypes = [wintypes.DWORD, ctypes.POINTER(GUITHREADINFO)]
_user32.GetGUIThreadInfo.restype = wintypes.BOOL
_user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
_user32.GetWindowLongW.restype = wintypes.LONG

# =============================================================================
# SECTION 2: UI AUTOMATION HELPERS
# =============================================================================

def _is_caps_lock_on():
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


# =============================================================================
# SECTION 3: MAIN HOOK MANAGER
# =============================================================================

class HookManager:
    """Manages the global keyboard hook and supporting listener threads.
    
    This class handles the state machine for the "Evaluation Pipeline". 
    It captures keystrokes, builds words, and triggers corrections.
    """

    def __init__(self, engine, sensitivity, blacklist, config):
        """Initialize the hook manager.

        Args:
            engine: EvaluationEngine instance for scoring words.
            sensitivity: SensitivityManager instance for dynamic thresholds.
            blacklist: BlacklistManager instance for app-specific disabling.
            config: Dict of configuration values from config.json.
        """
        self.engine = engine
        self.sensitivity = sensitivity
        self.blacklist = blacklist
        self.config = config

        self.enabled = config.get('enabled', True)
        self.debug_mode = config.get('debug_mode', False)
        self.idle_timeout = config.get('idle_timeout_seconds', 15.0)

        # Buffers for the current partial word
        self.buffer_active = ''  # What shows on screen right now
        self.buffer_shadow = ''  # The same keys translated to the other layout

        # Lookback queue: stores _WordEntry items for completed words in the
        # current session. Used for "Retroactive Correction" of ambiguous words.
        self.history_deque = collections.deque(maxlen=50)

        # Concurrency Lock: True when the switcher thread is erasing/re-injecting text.
        self.is_correcting = False
        # Buffer for keys the user types *while* the switch is happening.
        self.pending_queue = collections.deque(maxlen=100)

        self._hook_id = None
        self._hook_proc = None
        self._hook_thread = None
        self._mouse_listener = None
        self._running = False

        self._shift_pressed = False
        self._ctrl_pressed = False
        self._alt_pressed = False
        self._on_switch_callback = None

        # PERFORMANCE optimization: Cache expensive Windows API results.
        # These are updated on a slow 100ms polling thread.
        self._cached_layout = 'en'
        self._cached_blacklisted = False
        self._cached_is_ide_editor = False

        # Model Mode Selection: 'standard', 'smart', or 'technical'
        self.model_mode = config.get('model_mode', 'standard')

        # Suspension: user-configurable hotkey to temporarily disable the engine.
        # suspend_keybind is a frozenset of VK codes (e.g. {VK_CONTROL, 0x7B} for Ctrl+F12).
        # Empty set = feature not configured.
        self._suspend_keybind = frozenset(
            config.get('suspend_keybind_vks', [])
        )
        self._suspend_duration = config.get('suspend_duration_sec', 60)
        self._suspended_until = 0.0       # monotonic timestamp
        self._on_suspend_callback = None

    def set_enabled(self, enabled):
        """Enable or disable the engine logic (tray-accessible)."""
        self.enabled = enabled
        logger.info('Engine %s', 'enabled' if enabled else 'disabled')

    def set_debug_mode(self, enabled):
        """Enable or disable debug mode (expressive logging + CSV)."""
        self.debug_mode = enabled
        self.engine.set_enable_logging(enabled)
        logger.info('Debug Mode %s', 'enabled' if enabled else 'disabled')

    def set_on_switch_callback(self, callback):
        """Set a callback for when a layout switch occurs (e.g. for UI sounds)."""
        self._on_switch_callback = callback

    def set_on_suspend_callback(self, callback):
        """Set a callback for when the engine is suspended/resumed (e.g. for UI)."""
        self._on_suspend_callback = callback

    def set_suspend_config(self, keybind_vks, duration_sec):
        """Update the suspension hotkey and duration at runtime."""
        self._suspend_keybind = frozenset(keybind_vks)
        self._suspend_duration = duration_sec

    def set_model_mode(self, mode):
        """Update the model selection mode (standard, smart, technical)."""
        if mode in ('standard', 'smart', 'technical'):
            self.model_mode = mode
            logger.info('Model mode set to: %s', mode)
        else:
            logger.warning('Invalid model mode ignored: %r', mode)

    @property
    def is_suspended(self):
        """True if the engine is currently in a temporary suspension."""
        return time.monotonic() < self._suspended_until

    def _check_suspend_hotkey(self, vk):
        """Check if the current modifier+key state matches the suspend keybind.

        Returns:
            True if the hotkey was just triggered (and suspension toggled).
        """
        if not self._suspend_keybind:
            return False

        # Build the set of currently held keys
        currently_held = set()
        if self._ctrl_pressed:
            currently_held.add(VK_CONTROL)
        if self._shift_pressed:
            currently_held.add(VK_SHIFT)
        if self._alt_pressed:
            currently_held.add(VK_MENU)
        currently_held.add(vk)

        if currently_held == self._suspend_keybind:
            if self.is_suspended:
                # Cancel current suspension
                self._suspended_until = 0.0
                logger.info('Suspension cancelled by hotkey')
            else:
                self._suspended_until = time.monotonic() + self._suspend_duration
                self._clear_buffers()
                self._clear_history()
                logger.info('Engine suspended for %d seconds', self._suspend_duration)
            if self._on_suspend_callback:
                self._on_suspend_callback(self.is_suspended)
            return True
        return False

    def _set_correcting(self, val):
        """Internal lock flag setter used by the Switcher thread."""
        self.is_correcting = val

    def _clear_buffers(self):
        """Clear the current word buffers (usually on word completion or CRE)."""
        self.buffer_active = ''
        self.buffer_shadow = ''

    def _clear_history(self):
        """Clear the lookback history.
        
        CRITICAL: This is called on every Context Resumption Event (CRE) to
        prevent retroactive corrections from jumping across windows or clicks.
        """
        self.history_deque.clear()

    def _build_correction_block(self):
        """Collect contiguous ambiguous words from the recent history.
        
        This enables "delayed detection": if the engine realizes after 3 words
        that you've been in the wrong layout, it can fix all 3 words at once.

        Returns:
            List of _WordEntry items in chronological order.
        """
        block = []
        for entry in reversed(self.history_deque):
            if entry.is_ambiguous:
                block.append(entry)
            else:
                break
        block.reverse()
        if block:
            logger.debug(
                'Correction block: %d ambiguous words: %s',
                len(block), [e.active for e in block]
            )
        return block

    # -------------------------------------------------------------------------
    # PIPELINE CORE: HANDLE KEYPRESS
    # -------------------------------------------------------------------------

    def _handle_keypress(self, vk_code):
        """Process a single physical keypress through the evaluation pipeline.
        
        HOT PATH: This runs inside the OS low-level hook callback.
        Speed is critical. No I/O or slow API calls here.

        Args:
            vk_code: The virtual key code of the pressed key.

        Returns:
            True to BLOCK the key from reaching the target application.
            False to let the key pass through normally.
        """
        # 1. While a correction switch is happening, intercept and queue all typing.
        if self.is_correcting:
            caps_lock = _is_caps_lock_on()
            if vk_code in DELIMITER_VKS:
                self.pending_queue.append((vk_code, self._shift_pressed, caps_lock))
            else:
                en_ch, he_ch = get_both_chars(vk_code, self._shift_pressed, caps_lock)
                if en_ch is not None:
                    self.pending_queue.append((vk_code, self._shift_pressed, caps_lock))
            return True # Block keys so they don't interleave with correction injection.

        # 2. Skip logic if global disable, suspension, blacklist, or modifier combos (Ctrl+C)
        if not self.enabled:
            return False

        if self.is_suspended:
            return False

        if self._cached_blacklisted:
            return False

        if self._ctrl_pressed:
            if vk_code == VK_A:
                self.sensitivity.reset(reason='ctrl+a')
                self._clear_buffers()
                self._clear_history()
            return False

        if vk_code == VK_TAB:
            self.sensitivity.reset(reason='tab_key')
            self._clear_buffers()
            self._clear_history()
            return False

        # Reset sensitivity timer on every active keystroke
        self.sensitivity.record_keystroke()

        # 3. Handle Backspace (undo last buffer entry)
        if vk_code == VK_BACK:
            if self.buffer_active:
                self.buffer_active = self.buffer_active[:-1]
                self.buffer_shadow = self.buffer_shadow[:-1]
            elif self.history_deque:
                # Cross-boundary backspace: the user is deleting the delimiter
                # that separated the current (empty) word from the previous one.
                # Pop the previous word from history and restore it into the
                # buffers so lookback stays in sync with what's on screen.
                prev = self.history_deque.pop()
                self.buffer_active = prev.active
                self.buffer_shadow = prev.shadow
            return False

        # 4. Handle Delimiters (Space, Enter, Tab) — TRIGGER TIER 1 EVALUATION
        if vk_code in DELIMITER_VKS:
            delimiter_char = DELIMITER_CHARS.get(vk_code, ' ')

            if self.buffer_active:
                current = self._cached_layout
                
                # Determine effective evaluation mode
                eff_mode = self.model_mode
                if eff_mode == 'smart':
                    eff_mode = 'technical' if self._cached_is_ide_editor else 'standard'

                should_switch, diff, is_ambiguous = self.engine.evaluate(
                    self.buffer_active,
                    self.buffer_shadow,
                    self.sensitivity.delta,
                    current_layout=current,
                    on_delimiter=True,
                    mode=eff_mode
                )
                
                logger.debug(
                    'EVAL: "%s" (%s) -> "%s" | diff=%+.2f vs delta=%.2f | switch=%s | ambiguous=%s',
                    self.buffer_active, current, self.buffer_shadow, diff,
                    self.sensitivity.delta, should_switch, is_ambiguous
                )

                # Special Case: Hebrew + Caps Lock = Incorrect English.
                caps_lock = _is_caps_lock_on()
                needs_caps_fix = current == 'he' and caps_lock and not should_switch

                if should_switch or needs_caps_fix:
                    if self._trigger_switch(delimiter_char=delimiter_char, force_target=current if needs_caps_fix else None):
                        return True

                # Not a switch? Store word for potential future retroactive correction.
                self.history_deque.append(_WordEntry(
                    active=self.buffer_active,
                    shadow=self.buffer_shadow,
                    delimiter=delimiter_char,
                    is_ambiguous=is_ambiguous,
                ))

                self.sensitivity.on_word_complete()
            self._clear_buffers()
            return False

        # 5. Process Regular Character — TRIGGER TIER 2 EVALUATION (Mid-word)
        caps_lock = _is_caps_lock_on()
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

        # Only run mid-word scoring after 3+ characters to avoid false switches.
        if len(self.buffer_active) >= 3:
            # Determine effective evaluation mode
            eff_mode = self.model_mode
            if eff_mode == 'smart':
                eff_mode = 'technical' if self._cached_is_ide_editor else 'standard'

            should_switch, diff, is_ambiguous = self.engine.evaluate(
                self.buffer_active,
                self.buffer_shadow,
                self.sensitivity.delta,
                current_layout=current,
                mode=eff_mode
            )
            logger.debug(
                    'EVAL: "%s" (%s) -> "%s" | diff=%+.2f vs delta=%.2f | switch=%s | ambiguous=%s',
                    self.buffer_active, current, self.buffer_shadow, diff,
                    self.sensitivity.delta, should_switch, is_ambiguous
                )
            
            caps_lock = _is_caps_lock_on()
            needs_caps_fix = current == 'he' and caps_lock and not should_switch

            if should_switch or needs_caps_fix:
                if self._trigger_switch(force_target=current if needs_caps_fix else None):
                    return True

        return False

    def _trigger_switch(self, delimiter_char=None, force_target=None):
        """Prepare metadata and launch the async Switcher thread.
        
        Args:
            delimiter_char: Delimiter that triggered the switch, if any.
            force_target: Target layout override (used for Caps Lock fix).
            
        Returns:
            True if a switch sequence was officially started.
        """
        current = self._cached_layout
        target = force_target if force_target is not None else ('he' if current == 'en' else 'en')
        fix_caps = False

        # 1. Determine if we are just fixing Caps Lock within the same layout.
        caps_lock = _is_caps_lock_on()
        if current == 'he' and caps_lock:
            if target == 'en':
                # User intended English (HEB+CAPS=ENG). Per spec, we allow this manually.
                logger.debug('English intent detected in Hebrew layout with Caps Lock ON - ignoring switch')
                return False

            # User intended Hebrew but Caps was on. Stay in Hebrew, but fix Caps Lock.
            target = 'he'
            fix_caps = True

        # 2. Build the lookback correction block BEFORE clearing history.
        correction_block = self._build_correction_block()

        logger.info(
            'SWITCHING: "%s" -> "%s" (layout %s -> %s, fix_caps=%s) lookback=%d words',
            self.buffer_active, self.buffer_shadow, current, target,
            fix_caps, len(correction_block)
        )

        buf_active = self.buffer_active
        buf_shadow = self.buffer_shadow
        
        # 3. Clean up manager state before handing off to the thread.
        if delimiter_char is not None:
            self._clear_buffers()
        else:
            # Mid-word switch: Swap buffers to stay in sync with the new layout on screen.
            # (e.g. if we switched from HE to EN, what was 'shadow' is now 'active').
            self.buffer_active, self.buffer_shadow = self.buffer_shadow, self.buffer_active

        self._clear_history()

        if fix_caps:
            # Swap buffers so switcher erases the "bad" English and injects "intended" Hebrew.
            buf_active, buf_shadow = buf_shadow, buf_active
            correction_block = [
                _WordEntry(active=e.shadow, shadow=e.active, delimiter=e.delimiter, is_ambiguous=e.is_ambiguous)
                for e in correction_block
            ]

        # 4. Lock the main hook (synchronously) and spin up the heavy-lifter thread.
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
        """Background thread worker to execute the erase/toggle/inject sequence."""
        # Pre-emptive cache update: prevents hook from processing keys with old layout
        # before 'is_correcting' gets flipped back to False.
        self._cached_layout = target

        consumed_items = execute_switch(
            buf_active,
            buf_shadow,
            self.pending_queue,
            self._set_correcting,
            target,
            correction_block=correction_block,
            trigger_delimiter=trigger_delimiter,
            fix_caps=fix_caps,
        )

        # IMPORTANT: Integrate pending characters into the buffers so the engine
        # knows about the FULL word currently on screen.
        for q_vk, q_shift, q_caps in (consumed_items or []):
            # If the user typed a delimiter (Space/Enter) while the switch was
            # happening, it means the current word is finished. Clear buffers.
            if q_vk in DELIMITER_VKS:
                self._clear_buffers()
                continue

            en_ch, he_ch = get_both_chars(q_vk, q_shift, q_caps)
            if en_ch:
                if target == 'en':
                    self.buffer_active += en_ch
                    self.buffer_shadow += he_ch
                else:
                    self.buffer_active += he_ch
                    self.buffer_shadow += en_ch

        self.sensitivity.reset(reason='layout_switch')

        if self._on_switch_callback:
            self._on_switch_callback()

    # -------------------------------------------------------------------------
    # SECTION 4: LOW-LEVEL HOOK ENGINE (Message Pumps & Threads)
    # -------------------------------------------------------------------------

    def _kb_hook_callback(self, n_code, w_param, l_param):
        """The core WH_KEYBOARD_LL callback function.
        
        This is the most time-sensitive block of code in the entire application.
        Failure to call NextHookEx quickly enough will lag the whole OS.
        """
        try:
            if n_code == HC_ACTION:
                kb = ctypes.cast(
                    l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)
                ).contents

                vk = kb.vkCode

                # Important: Do not re-process keys we injected ourselves.
                if kb.flags & LLKHF_INJECTED:
                    return _user32.CallNextHookEx(
                        self._hook_id, n_code, w_param, l_param
                    )

                if w_param in (WM_KEYDOWN, WM_SYSKEYDOWN):
                    # Track modifier states for combo-blocking
                    if vk in (VK_SHIFT, VK_LSHIFT, VK_RSHIFT):
                        self._shift_pressed = True
                    elif vk in (VK_CONTROL, VK_LCONTROL, VK_RCONTROL):
                        self._ctrl_pressed = True
                    elif vk in (VK_MENU, VK_LMENU, VK_RMENU):
                        self._alt_pressed = True
                    elif vk in MODIFIER_VKS:
                        pass
                    else:
                        # Check suspend hotkey before normal processing
                        if self._check_suspend_hotkey(vk):
                            pass  # Let the key through, we just toggled suspension
                        else:
                            block = self._handle_keypress(vk)
                            if block:
                                return 1 # Stop the key from reaching the original app.
                else:
                    # KEY UP events
                    if vk in (VK_SHIFT, VK_LSHIFT, VK_RSHIFT):
                        self._shift_pressed = False
                    elif vk in (VK_CONTROL, VK_LCONTROL, VK_RCONTROL):
                        self._ctrl_pressed = False
                    elif vk in (VK_MENU, VK_LMENU, VK_RMENU):
                        self._alt_pressed = False
        except Exception:
            logger.exception('Error in keyboard hook callback')

        # Always pass control to the next hook in the chain unless explicitly blocked.
        return _user32.CallNextHookEx(
            self._hook_id, n_code, w_param, l_param
        )

    def _hook_thread_func(self):
        """Thread worker that installs the hook and maintains the Windows message pump."""
        self._hook_proc = HOOKPROC(self._kb_hook_callback)

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
            logger.error('Failed to install keyboard hook, error=%d', err)
            return

        logger.info('Keyboard hook installed (id=%s)', self._hook_id)

        # The message pump is required for low-level hooks to receive events.
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
        """Mouse click callback — triggers a Context Resumption Event (CRE)."""
        if pressed and button != pynput_mouse.Button.middle:
            self.sensitivity.reset(reason='mouse_click')
            self._clear_buffers()
            self._clear_history()

    def _poll_foreground_window(self):
        """Worker thread to poll OS state (Layout/Blacklist) every 100ms.
        
        This moves slow Windows API calls OUT of the keyboard hook callback
        to maintain system responsiveness.
        """
        while self._running:
            try:
                # 1. Update Layout (Skip if mid-switch correction is happening)
                if not self.is_correcting:
                    new_layout = get_current_layout()
                    if new_layout != 'unknown':
                        if new_layout != self._cached_layout:
                            # Manual change detected? Trigger CRE.
                            self.sensitivity.reset(reason='manual_layout_change')
                            self._clear_buffers()
                            self._clear_history()
                        self._cached_layout = new_layout

                # 2. Update Blacklist and IDE Status
                exe = self.blacklist.get_foreground_exe()
                is_blacklisted = exe in self.blacklist.blacklisted
                if not is_blacklisted:
                    try:
                        is_blacklisted = is_password_field_active()
                    except Exception as e:
                        logger.debug('Error checking password field: %s', e)
                self._cached_blacklisted = is_blacklisted
                self._cached_is_ide_editor = self.blacklist.is_ide_editor(exe)

                # 3. Detect Foreground Window changes (Trigger CRE)
                hwnd = user32.GetForegroundWindow()
                if self.sensitivity.check_window_change(hwnd):
                    self.sensitivity.reset(reason='window_change')
                    self._clear_buffers()
                    self._clear_history()

                # 4. Check for Idle Timeout (Trigger CRE)
                if self.enabled and self.sensitivity.check_idle_timeout(self.idle_timeout):
                    self.sensitivity.reset(reason='idle_timeout')
                    self.sensitivity.record_keystroke() # Prevent infinite reset loop
                    self._clear_buffers()
                    self._clear_history()

            except Exception:
                logger.exception('Error in foreground poll')

            time.sleep(0.1)

    def start(self):
        """Launch all listener threads (Keyboard, Mouse, OS Polling)."""
        self._running = True

        self._cached_layout = get_current_layout()
        logger.info('Initial layout: %s', self._cached_layout)

        # Thread 1: The hot keyboard hook
        self._hook_thread = threading.Thread(
            target=self._hook_thread_func,
            daemon=True,
            name='KeyboardHookThread'
        )
        self._hook_thread.start()

        # Thread 2: Mouse listener (pynput)
        self._mouse_listener = pynput_mouse.Listener(
            on_click=self._on_mouse_click
        )
        self._mouse_listener.daemon = True
        self._mouse_listener.start()

        # Thread 3: Slow poller for OS/UI context
        self._fg_thread = threading.Thread(
            target=self._poll_foreground_window,
            daemon=True,
            name='ForegroundPollThread'
        )
        self._fg_thread.start()
        logger.info('All hooks and polling threads started')

    def stop(self):
        """Safely dismantle all hooks and stop background threads."""
        self._running = False

        if self._hook_id and self._hook_thread and self._hook_thread.ident:
            # Send WM_QUIT to the hook thread's message pump
            _user32.PostThreadMessageW(
                self._hook_thread.ident,
                WM_QUIT,
                0, 0
            )

        if self._mouse_listener:
            self._mouse_listener.stop()

        if self._hook_thread and self._hook_thread.is_alive():
            self._hook_thread.join(timeout=2.0)

        logger.info('All hooks stopped')
