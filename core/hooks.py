"""
hooks.py — Global keyboard hook and evaluation pipeline.

This module is the "sensory system" of SwitchLang. It delegates all
OS-specific keyboard interception to a PlatformBackend, then processes
keystrokes through the N-gram evaluation pipeline and triggers layout
corrections when gibberish is detected.

KEY CONCEPTS:
- Low-Level Hook: Intercepts keys before the OS or other apps see them.
  (Implemented by the platform backend: WH_KEYBOARD_LL on Windows,
   evdev.grab() on Linux.)
- Shadow Buffer: Keeps track of what the keys WOULD be in the other layout.
- Context Resumption Event (CRE): Resets buffers and sensitivity when the
  user switches windows, clicks the mouse, or stays idle, signaling a "fresh start".
"""

import collections
import logging
import threading
import time

from core.keymap import get_both_chars
from core.switcher import execute_switch

logger = logging.getLogger('switchlang.hooks')

# =============================================================================
# VIRTUAL KEY CONSTANTS (platform-normalised, Windows VK_* convention)
# =============================================================================
# These are used by the evaluation pipeline regardless of OS.
# The platform backend translates native keycodes to these values.

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
    '_WordEntry', ['active', 'shadow', 'delimiter', 'is_colliding', 'is_ambiguous']
)


# =============================================================================
# MAIN HOOK MANAGER
# =============================================================================


class HookManager:
    """Manages the global keyboard hook and supporting listener threads.

    This class handles the state machine for the "Evaluation Pipeline".
    It captures keystrokes, builds words, and triggers corrections.
    All OS-specific I/O is delegated to the platform backend.
    """

    def __init__(self, engine, sensitivity, blacklist, config, platform):
        """Initialize the hook manager.

        Args:
            engine: EvaluationEngine instance for scoring words.
            sensitivity: SensitivityManager instance for dynamic thresholds.
            blacklist: BlacklistManager instance for app-specific disabling.
            config: Dict of configuration values from config.json.
            platform: PlatformBackend instance for OS-specific operations.
        """
        self.engine = engine
        self.sensitivity = sensitivity
        self.blacklist = blacklist
        self.config = config
        self.platform = platform

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

        self._running = False

        self._shift_pressed = False
        self._ctrl_pressed = False
        self._alt_pressed = False
        self._on_switch_callback = None

        # PERFORMANCE optimization: Cache expensive OS API results.
        # These are updated on a slow 100ms polling thread.
        self._cached_layout = 'en'
        self._cached_blacklisted = False
        self._cached_is_ide_editor = False

        # Model Mode Selection: 'standard', 'smart', or 'technical'
        self.model_mode = config.get('model_mode', 'standard')

        # Suspension: user-configurable hotkey to temporarily disable the engine.
        self._suspend_keybind = frozenset(
            config.get('suspend_keybind_vks', [])
        )
        self._suspend_duration = config.get('suspend_duration_sec', 60)
        self._suspended_until = 0.0
        self._on_suspend_callback = None

        # Track last known foreground window for CRE detection
        self._last_foreground_hwnd = None

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
        """Collect contiguous correctable words from the recent history.

        A word is correctable if it is either a collision (is_colliding) or
        ambiguous (is_ambiguous). This enables "delayed detection": if the
        engine realizes after 3 words that you've been in the wrong layout,
        it can fix all 3 words at once.

        Returns:
            List of _WordEntry items in chronological order.
        """
        block = []
        for entry in reversed(self.history_deque):
            if entry.is_colliding or entry.is_ambiguous:
                block.append(entry)
            else:
                break
        block.reverse()
        if block:
            logger.debug(
                'Correction block: %d correctable words: %s',
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
            vk_code: The normalised virtual key code of the pressed key.

        Returns:
            True to BLOCK the key from reaching the target application.
            False to let the key pass through normally.
        """
        # 1. While a correction switch is happening, intercept and queue all typing.
        if self.is_correcting:
            caps_lock = self.platform.is_caps_lock_on()
            if vk_code in DELIMITER_VKS:
                self.pending_queue.append((vk_code, self._shift_pressed, caps_lock))
            else:
                en_ch, he_ch = get_both_chars(vk_code, self._shift_pressed, caps_lock)
                if en_ch is not None:
                    self.pending_queue.append((vk_code, self._shift_pressed, caps_lock))
            return True  # Block keys so they don't interleave with correction injection.

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

                should_switch, diff, is_colliding, is_ambiguous = self.engine.evaluate(
                    self.buffer_active,
                    self.buffer_shadow,
                    self.sensitivity.delta,
                    current_layout=current,
                    on_delimiter=True,
                    mode=eff_mode
                )

                logger.debug(
                    'EVAL: "%s" (%s) -> "%s" | diff=%+.2f vs delta=%.2f | switch=%s | colliding=%s | ambiguous=%s',
                    self.buffer_active, current, self.buffer_shadow, diff,
                    self.sensitivity.delta, should_switch, is_colliding, is_ambiguous
                )

                # Special Case: Hebrew + Caps Lock = Incorrect English.
                caps_lock = self.platform.is_caps_lock_on()
                needs_caps_fix = current == 'he' and caps_lock and not should_switch

                if should_switch or needs_caps_fix:
                    if self._trigger_switch(delimiter_char=delimiter_char, force_target=current if needs_caps_fix else None):
                        return True

                # Not a switch? Store word for potential future retroactive correction.
                self.history_deque.append(_WordEntry(
                    active=self.buffer_active,
                    shadow=self.buffer_shadow,
                    delimiter=delimiter_char,
                    is_colliding=is_colliding,
                    is_ambiguous=is_ambiguous,
                ))

                self.sensitivity.on_word_complete()
            self._clear_buffers()
            return False

        # 5. Process Regular Character — TRIGGER TIER 2 EVALUATION (Mid-word)
        caps_lock = self.platform.is_caps_lock_on()
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
            eff_mode = self.model_mode
            if eff_mode == 'smart':
                eff_mode = 'technical' if self._cached_is_ide_editor else 'standard'

            should_switch, diff, is_colliding, is_ambiguous = self.engine.evaluate(
                self.buffer_active,
                self.buffer_shadow,
                self.sensitivity.delta,
                current_layout=current,
                mode=eff_mode
            )
            logger.debug(
                    'EVAL: "%s" (%s) -> "%s" | diff=%+.2f vs delta=%.2f | switch=%s | colliding=%s | ambiguous=%s',
                    self.buffer_active, current, self.buffer_shadow, diff,
                    self.sensitivity.delta, should_switch, is_colliding, is_ambiguous
                )

            caps_lock = self.platform.is_caps_lock_on()
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
        caps_lock = self.platform.is_caps_lock_on()
        if current == 'he' and caps_lock:
            if target == 'en':
                logger.debug('English intent detected in Hebrew layout with Caps Lock ON - ignoring switch')
                return False

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
            self.buffer_active, self.buffer_shadow = self.buffer_shadow, self.buffer_active

        self._clear_history()

        if fix_caps:
            buf_active, buf_shadow = buf_shadow, buf_active
            correction_block = [
                _WordEntry(active=e.shadow, shadow=e.active, delimiter=e.delimiter, is_colliding=e.is_colliding, is_ambiguous=e.is_ambiguous)
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
        # Pre-emptive cache update
        self._cached_layout = target

        consumed_items = execute_switch(
            buf_active,
            buf_shadow,
            self.pending_queue,
            self._set_correcting,
            target,
            self.platform,
            correction_block=correction_block,
            trigger_delimiter=trigger_delimiter,
            fix_caps=fix_caps,
        )

        # Integrate pending characters into the buffers
        for q_vk, q_shift, q_caps in (consumed_items or []):
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
    # KEY EVENT CALLBACKS (called by the platform backend)
    # -------------------------------------------------------------------------

    def _on_key_down(self, vk, shifted):
        """Key-down callback dispatched from the platform backend.

        Args:
            vk: Normalised virtual key code.
            shifted: Whether Shift is currently held.

        Returns:
            True to BLOCK the key, False to let it through.
        """
        # Track modifier states
        if vk in (VK_SHIFT, VK_LSHIFT, VK_RSHIFT):
            self._shift_pressed = True
            return False
        elif vk in (VK_CONTROL, VK_LCONTROL, VK_RCONTROL):
            self._ctrl_pressed = True
            return False
        elif vk in (VK_MENU, VK_LMENU, VK_RMENU):
            self._alt_pressed = True
            return False
        elif vk in MODIFIER_VKS:
            return False

        # Use the shifted state from the backend (more reliable on Linux)
        self._shift_pressed = shifted

        # Check suspend hotkey before normal processing
        if self._check_suspend_hotkey(vk):
            return False  # Let the key through, we just toggled suspension

        return self._handle_keypress(vk)

    def _on_key_up(self, vk):
        """Key-up callback dispatched from the platform backend.

        Args:
            vk: Normalised virtual key code.
        """
        if vk in (VK_SHIFT, VK_LSHIFT, VK_RSHIFT):
            self._shift_pressed = False
        elif vk in (VK_CONTROL, VK_LCONTROL, VK_RCONTROL):
            self._ctrl_pressed = False
        elif vk in (VK_MENU, VK_LMENU, VK_RMENU):
            self._alt_pressed = False

    # -------------------------------------------------------------------------
    # FOREGROUND POLLING & CONTEXT DETECTION
    # -------------------------------------------------------------------------

    def _on_mouse_click(self, x, y, button, pressed):
        """Mouse click callback — triggers a Context Resumption Event (CRE)."""
        from pynput import mouse as pynput_mouse
        if pressed and button != pynput_mouse.Button.middle:
            self.sensitivity.reset(reason='mouse_click')
            self._clear_buffers()
            self._clear_history()

    def _poll_foreground_window(self):
        """Worker thread to poll OS state (Layout/Blacklist) every 100ms.

        This moves slow OS API calls OUT of the keyboard hook callback
        to maintain system responsiveness.
        """
        while self._running:
            try:
                # 1. Update Layout (Skip if mid-switch correction is happening)
                if not self.is_correcting:
                    new_layout = self.platform.get_current_layout()
                    if new_layout != 'unknown':
                        if new_layout != self._cached_layout:
                            # Manual change detected? Trigger CRE.
                            self.sensitivity.reset(reason='manual_layout_change')
                            self._clear_buffers()
                            self._clear_history()
                        self._cached_layout = new_layout

                # 2. Update Blacklist and IDE Status
                exe = self.platform.get_foreground_process()
                is_blacklisted = exe in self.blacklist.blacklisted
                if not is_blacklisted:
                    try:
                        is_blacklisted = self.platform.is_password_field_active()
                    except Exception as e:
                        logger.debug('Error checking password field: %s', e)
                self._cached_blacklisted = is_blacklisted
                self._cached_is_ide_editor = self.blacklist.is_ide_editor(exe)

                # 3. Detect Foreground Window changes (Trigger CRE)
                current_exe = exe  # Use process name as window identity proxy
                if current_exe != self._last_foreground_hwnd:
                    if self._last_foreground_hwnd is not None:
                        self.sensitivity.reset(reason='window_change')
                        self._clear_buffers()
                        self._clear_history()
                    self._last_foreground_hwnd = current_exe

                # 4. Check for Idle Timeout (Trigger CRE)
                if self.enabled and self.sensitivity.check_idle_timeout(self.idle_timeout):
                    self.sensitivity.reset(reason='idle_timeout')
                    self.sensitivity.record_keystroke()
                    self._clear_buffers()
                    self._clear_history()

            except Exception:
                logger.exception('Error in foreground poll')

            time.sleep(0.1)

    def start(self):
        """Launch all listener threads (Keyboard, Mouse, OS Polling)."""
        self._running = True

        self._cached_layout = self.platform.get_current_layout()
        logger.info('Initial layout: %s', self._cached_layout)

        # Thread 1: Keyboard hook (platform-specific)
        self.platform.start_keyboard_hook(self._on_key_down, self._on_key_up)

        # Thread 2: Mouse listener (platform-specific)
        self.platform.start_mouse_listener(self._on_mouse_click)

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

        self.platform.stop_keyboard_hook()
        self.platform.stop_mouse_listener()

        logger.info('All hooks stopped')
