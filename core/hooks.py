"""
hooks.py — Keystroke evaluation pipeline and context-resumption logic.

This module is the "sensory system" of SwitchLang. It receives every physical
keystroke from the OS layer (core/os_*.py, via core.osapi), processes it through
an N-gram evaluation pipeline, and triggers layout corrections when gibberish
is detected.

KEY CONCEPTS:
- OS hook: Lives in os_windows / os_linux / os_macos; reports each key to
  HookManager._on_key before the OS or other apps see it.
- Shadow Buffer: Keeps track of what the keys WOULD be in the other layout.
- Context Resumption Event (CRE): Resets buffers and sensitivity when the
  user switches windows, clicks the mouse, or stays idle, signaling a "fresh start".
"""

import collections
import logging
import threading
import time

from core import osapi
from core.keymap import get_both_chars, vk_to_char
from core.switcher import execute_switch

logger = logging.getLogger('switchlang.hooks')


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
VK_LWIN = 0x5B   # Super / Windows key
VK_RWIN = 0x5C
VK_RETURN = 0x0D
VK_SPACE = 0x20
VK_BACK = 0x08
VK_TAB = 0x09
VK_CAPITAL = 0x14
VK_A = 0x41   # Used for Ctrl+A CRE


# Delimiters trigger word-level evaluation
DELIMITER_VKS = {VK_SPACE, VK_RETURN}
DELIMITER_CHARS = {VK_SPACE: ' ', VK_RETURN: '\n'}

# Stores a completed word and its metadata for retroactive correction (Lookback)
_WordEntry = collections.namedtuple(
    '_WordEntry', ['active', 'shadow', 'delimiter', 'is_colliding', 'is_ambiguous', 'evidence']
)


# =============================================================================
# SECTION 1: MAIN HOOK MANAGER
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
        self._consecutive_midword_hits = 0  # K=2 consecutive mid-word confirmation counter

        # Lookback queue: stores _WordEntry items for completed words in the
        # current session. Used for "Retroactive Correction" of ambiguous words.
        self.history_deque = collections.deque(maxlen=50)

        # Concurrency Lock: True when the switcher thread is erasing/re-injecting text.
        self.is_correcting = False
        # Buffer for keys the user types *while* the switch is happening.
        self.pending_queue = collections.deque()
        self._context_lock = threading.RLock()
        self._input_context = None
        self._context_generation = 0
        self._sample_sequence = 0
        self._published_sequence = 0
        self._correction_generation = None
        self._correction_target = None

        self._running = False

        self._shift_pressed = False
        self._ctrl_pressed = False
        self._alt_pressed = False
        self._win_pressed = False
        self._letter_shift_history = []

        # PERFORMANCE optimization: Cache expensive Windows API results.
        # These are updated on a slow 100ms polling thread.
        self._cached_layout = 'en'
        self._cached_blacklisted = False
        self._cached_is_ide_editor = False

        # Model Mode Selection: 'standard', 'smart', or 'technical'
        self.model_mode = config.get('model_mode', 'standard')

        # Suspension hotkey (frozenset of VK codes, e.g. Ctrl+F12, or empty if disabled).
        self._suspend_keybind = frozenset(
            config.get('suspend_keybind_vks', [])
        )
        self._suspend_duration = config.get('suspend_duration_sec', 60)
        self._suspend_switch_layout = config.get('suspend_switch_layout', False)
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


    def set_on_suspend_callback(self, callback):
        """Set a callback for when the engine is suspended/resumed (e.g. for UI)."""
        self._on_suspend_callback = callback

    def set_suspend_config(self, keybind_vks, duration_sec, switch_layout=False):
        """Update the suspension hotkey and duration at runtime."""
        self._suspend_keybind = frozenset(keybind_vks)
        self._suspend_duration = duration_sec
        self._suspend_switch_layout = switch_layout

    def set_model_mode(self, mode):
        """Update the model selection mode (standard, smart, technical)."""
        if mode in ('standard', 'smart', 'technical'):
            self.model_mode = mode
            self.engine.set_model_mode(mode)
            logger.info('Model mode set to: %s', mode)
        else:
            logger.warning('Invalid model mode ignored: %r', mode)

    def apply_config(self, config_data):
        """Apply new configuration values at runtime."""
        self.set_enabled(config_data.get('enabled', True))
        self.set_debug_mode(config_data.get('debug_mode', False))
        self.idle_timeout = config_data.get('idle_timeout_seconds', 15.0)
        self.sensitivity.update_config(
            baseline_delta=config_data.get('baseline_delta', 6.0),
            alpha=config_data.get('sensitivity_alpha', 0.1)
        )
        self.set_suspend_config(
            config_data.get('suspend_keybind_vks', []),
            config_data.get('suspend_duration_sec', 60),
            config_data.get('suspend_switch_layout', False)
        )
        self.set_model_mode(config_data.get('model_mode', 'standard'))

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
                self._fire_cre('engine_suspend')

                logger.info('Engine suspended for %d seconds', self._suspend_duration)

                if self._suspend_switch_layout:
                    target = 'he' if self._cached_layout == 'en' else 'en'
                    self._cached_layout = target
                    threading.Thread(target=osapi.toggle_layout, args=(target,), daemon=True).start()
                    logger.info('Layout switched to %s on suspend', target)

            if self._on_suspend_callback:
                self._on_suspend_callback(self.is_suspended)
            return True
        return False

    def _fire_cre(self, reason):
        """Context Resumption Event — reset sensitivity and all buffers."""
        with self._context_lock:
            self._context_generation += 1
            self.sensitivity.reset(reason=reason)
            self._clear_buffers()
            self._clear_history()

    @property
    def correction_status(self):
        if self._input_context is None:
            return ('Auto-correction paused: active layout unavailable. '
                    'On GNOME, turn off per-window input sources.')
        return ''

    def _refresh_input_context(self):
        """Publish one coherent sample; a context loss cancels old corrections."""
        with self._context_lock:
            self._sample_sequence += 1
            sequence = self._sample_sequence
        # GNOME runs subprocesses here; never hold the keyboard's state lock.
        context = osapi.get_input_context()
        with self._context_lock:
            if sequence < self._published_sequence:
                # Superseded is not the same as unavailable. The newer sample
                # and its generation decide whether a correction is still valid.
                return self._input_context
            self._published_sequence = sequence
            if context is not None and context.layout not in ('en', 'he'):
                context = None
            previous = self._input_context
            if context != previous:
                own_layout_change = (
                    self.is_correcting and context is not None and previous is not None
                    and context.identity == previous.identity
                    and context.layout == self._correction_target
                )
                if not own_layout_change:
                    self._fire_cre('input_context_change')
            elif context is None and (self.buffer_active or self.history_deque):
                self._fire_cre('input_context_unavailable')
            self._input_context = context
            self._cached_layout = context.layout if context else 'unknown'
            return context


    def _clear_buffers(self):
        """Clear the current word buffers (usually on word completion or CRE)."""
        self.buffer_active = ''
        self.buffer_shadow = ''
        self._consecutive_midword_hits = 0
        self._letter_shift_history.clear()

    def _is_all_shift_word(self):
        """True if all letters in the current word were typed with Shift (and Caps Lock is off)."""
        return bool(
            not osapi.is_caps_lock_on()
            and self._letter_shift_history
            and all(s for _, s in self._letter_shift_history)
        )

    def _clear_history(self):
        """Clear the lookback history.
        
        CRITICAL: This is called on every Context Resumption Event (CRE) to
        prevent retroactive corrections from jumping across windows or clicks.
        """
        self.history_deque.clear()

    def _evidence(self):
        """Switch evidence carried from previous words; history clears reset it to 0."""
        return self.history_deque[-1].evidence if self.history_deque else 0.0

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

    def _evaluate_current(self, effective_layout, on_delimiter=False):
        """Evaluate current buffers and log decision details."""
        # 'smart' resolves to technical inside IDE editors and standard elsewhere;
        # it must never reach the engine, whose mode is 'standard' or 'technical'.
        eff_mode = self.model_mode
        if eff_mode == 'smart':
            eff_mode = 'technical' if self._cached_is_ide_editor else 'standard'
        should_switch, diff, is_colliding, is_ambiguous = self.engine.evaluate(
            self.buffer_active,
            self.buffer_shadow,
            self.sensitivity.delta,
            current_layout=effective_layout,
            on_delimiter=on_delimiter,
            mode=eff_mode,
            evidence=self._evidence(),
        )
        logger.debug(
            'EVAL: "%s" (%s/%s) -> "%s" | diff=%+.2f evidence=%.2f vs delta=%.2f | switch=%s | colliding=%s | ambiguous=%s',
            self.buffer_active, self._cached_layout, effective_layout, self.buffer_shadow, diff,
            self._evidence(), self.sensitivity.delta, should_switch, is_colliding, is_ambiguous
        )
        return should_switch, diff, is_colliding, is_ambiguous

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
            if self._correction_generation != self._context_generation:
                return False
            caps_lock = osapi.is_caps_lock_on()
            self.pending_queue.append((vk_code, self._shift_pressed, caps_lock))
            return True # Block keys so they don't interleave with correction injection.

        # 2. Skip logic if global disable, suspension, blacklist, or modifier combos (Ctrl+C)
        if not self.enabled:
            return False

        if self.is_suspended:
            return False

        if self._cached_blacklisted:
            return False

        if self._win_pressed:  # Super+key is a desktop shortcut, not typing
            return False

        if self._ctrl_pressed:
            if vk_code == VK_A:
                self._fire_cre('ctrl+a')

            return False

        if vk_code == VK_TAB:
            self._fire_cre('tab_key')

            return False

        # Caps Lock toggle: treat as CRE to avoid mixed buffers.
        if vk_code == VK_CAPITAL:
            self._fire_cre('caps_lock')

            return False

        # Reset sensitivity timer on every active keystroke
        self.sensitivity.record_keystroke()

        # 3. Handle Backspace (undo last buffer entry)
        if vk_code == VK_BACK:
            self._consecutive_midword_hits = 0
            if self.buffer_active:
                self.buffer_active = self.buffer_active[:-1]
                self.buffer_shadow = self.buffer_shadow[:-1]
                while self._letter_shift_history and self._letter_shift_history[-1][0] >= len(self.buffer_active):
                    self._letter_shift_history.pop()
            elif self.history_deque:
                # Cross-boundary backspace: the user is deleting the delimiter
                # that separated the current (empty) word from the previous one.
                last = self.history_deque[-1]
                if len(last.delimiter) > 1:
                    # Multi-character delimiter (e.g. Space + Enter). 
                    # Just erase the last delimiter character.
                    self.history_deque[-1] = last._replace(delimiter=last.delimiter[:-1])
                else:
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

                # In Hebrew layout, typing an English word using Shift is intentional.
                if current == 'he' and self._is_all_shift_word():
                    self._clear_buffers()
                    self.sensitivity.on_word_complete()
                    return False

                # When Hebrew layout + Caps Lock, screen shows English.
                # Buffers already reflect this (set in step 5), so evaluate as English.
                caps_lock = osapi.is_caps_lock_on()
                effective_layout = 'en' if (current == 'he' and caps_lock) else current

                should_switch, diff, is_colliding, is_ambiguous = self._evaluate_current(
                    effective_layout, on_delimiter=True
                )

                if should_switch:
                    is_caps_fix = (current == 'he' and caps_lock)
                    if self._trigger_switch(delimiter_char=delimiter_char, caps_fix=is_caps_fix):
                        return True

                # Not a switch? Store word for potential future retroactive correction.
                self.history_deque.append(_WordEntry(
                    active=self.buffer_active,
                    shadow=self.buffer_shadow,
                    delimiter=delimiter_char,
                    is_colliding=is_colliding,
                    is_ambiguous=is_ambiguous,
                    evidence=self.engine.accumulate(self._evidence(), diff, is_colliding),
                ))

                self.sensitivity.on_word_complete()
            elif self.history_deque:
                # The user typed a delimiter on an empty buffer.
                # This usually means consecutive delimiters (e.g., Space then Enter).
                # Append this delimiter to the last word's delimiter string so it's erased properly.
                last = self.history_deque.pop()
                self.history_deque.append(last._replace(delimiter=last.delimiter + delimiter_char))
                
            self._clear_buffers()
            return False

        # 5. Process Regular Character — TRIGGER TIER 2 EVALUATION (Mid-word)
        caps_lock = osapi.is_caps_lock_on()
        en_char, he_char = get_both_chars(vk_code, self._shift_pressed, caps_lock)
        if en_char is None:
            return False

        if 0x41 <= vk_code <= 0x5A:
            self._letter_shift_history.append((len(self.buffer_active), self._shift_pressed))

        current = self._cached_layout

        # When Hebrew layout + Caps Lock, Windows outputs English on screen.
        # Treat the effective layout as English so buffers match reality.
        effective_layout = 'en' if (current == 'he' and caps_lock) else current

        if effective_layout == 'en':
            self.buffer_active += en_char
            self.buffer_shadow += he_char
        else:
            self.buffer_active += he_char
            self.buffer_shadow += en_char

        # Only run mid-word scoring after 3+ characters to avoid false switches.
        if len(self.buffer_active) >= 3:
            if current == 'he' and self._is_all_shift_word():
                return False

            should_switch, diff, is_colliding, is_ambiguous = self._evaluate_current(
                effective_layout, on_delimiter=False
            )

            if should_switch:
                self._consecutive_midword_hits += 1
                if self._consecutive_midword_hits >= 2:
                    is_caps_fix = (current == 'he' and caps_lock)
                    if self._trigger_switch(caps_fix=is_caps_fix):
                        return True
            else:
                self._consecutive_midword_hits = 0

        return False

    def _trigger_switch(self, delimiter_char=None, caps_fix=False):
        """Prepare metadata and launch the async Switcher thread.
        
        Args:
            delimiter_char: Delimiter that triggered the switch, if any.
            caps_fix: If True, this is a Caps Lock correction (HE layout + Caps ON,
                      user intended Hebrew). Stay in HE, toggle Caps Lock off.
            
        Returns:
            True if a switch sequence was officially started.
        """
        current = self._cached_layout

        if caps_fix:
            # HE+CapsLock: user meant Hebrew. Stay in Hebrew, just fix Caps Lock.
            target = 'he'
        else:
            target = 'he' if current == 'en' else 'en'

        # Build the lookback correction block BEFORE clearing history.
        correction_block = self._build_correction_block()

        logger.info(
            'SWITCHING: "%s" -> "%s" (layout %s -> %s, caps_fix=%s) lookback=%d words',
            self.buffer_active, self.buffer_shadow, current, target,
            caps_fix, len(correction_block)
        )

        buf_active = self.buffer_active
        buf_shadow = self.buffer_shadow
        
        # Clean up manager state before handing off to the thread.
        if delimiter_char is not None:
            self._clear_buffers()
        else:
            # Mid-word switch: Swap buffers to stay in sync with the new layout on screen.
            # (e.g. if we switched from HE to EN, what was 'shadow' is now 'active').
            self.buffer_active, self.buffer_shadow = self.buffer_shadow, self.buffer_active
            self._consecutive_midword_hits = 0

        self._clear_history()

        # Lock the main hook (synchronously) and spin up the heavy-lifter thread.
        self._correction_generation = self._context_generation
        self._correction_target = target
        self.is_correcting = True

        switch_thread = threading.Thread(
            target=self._do_switch,
            args=(buf_active, buf_shadow, target, correction_block, delimiter_char, caps_fix),
            daemon=True,
            name='SwitchThread'
        )
        switch_thread.start()
        return True

    def _do_switch(self, buf_active, buf_shadow, target,
                   correction_block=None, trigger_delimiter=None,
                   caps_fix=False):
        """Background thread worker to execute the erase/toggle/inject sequence.

        Owns the full is_correcting lock lifecycle: the lock is held from
        before the thread starts (_trigger_switch) until ALL post-correction
        work (pending queue drain + buffer integration) is complete. This
        prevents the hook thread from racing on buffer_active/buffer_shadow.
        """
        completed = False
        try:
            actual_layout = execute_switch(
                buf_active, buf_shadow, target,
                correction_block=correction_block,
                trigger_delimiter=trigger_delimiter,
                fix_caps=caps_fix,
                context_is_current=self._correction_is_current,
            )
            if actual_layout is None:
                return

            with self._context_lock:
                if actual_layout != target:
                    self._clear_buffers()
                    self._clear_history()

            # Pop, replay and integrate one key while holding the state lock.
            # Releasing correction ownership on an empty queue must be atomic
            # with that check, or a last key could be queued after the drain.
            while True:
                if not self._correction_is_current():
                    return
                with self._context_lock:
                    if self._correction_generation != self._context_generation:
                        return
                    if not self.pending_queue:
                        self.sensitivity.reset(reason='layout_switch')
                        self._correction_generation = None
                        self._correction_target = None
                        self.is_correcting = False
                        completed = True
                        return
                    q_vk, q_shift, q_caps = self.pending_queue.popleft()
                    ch = vk_to_char(q_vk, q_shift, layout=actual_layout, caps_lock=q_caps)
                    if actual_layout != target or not ch:
                        osapi.send_vk_key(q_vk, shift=q_shift)
                    else:
                        osapi.send_string_as_keys(ch, actual_layout)
                    if actual_layout == target:
                        self._integrate_pending_key(q_vk, q_shift, q_caps, actual_layout)
        except Exception:
            logger.exception('Error in switch thread')
        finally:
            if not completed:
                with self._context_lock:
                    # These keys were captured for the old context. Replaying
                    # them into whichever window now has focus is unsafe.
                    if self.pending_queue:
                        logger.warning('Correction cancelled; discarding %d old-context queued keys',
                                       len(self.pending_queue))
                    self.pending_queue.clear()
                    self._fire_cre('correction_cancelled')
                    self.is_correcting = False
                    self._correction_generation = None
                    self._correction_target = None

    def _correction_is_current(self):
        if not self._running or not self.enabled or self.is_suspended:
            return False
        context = self._refresh_input_context()
        with self._context_lock:
            return (context is not None
                    and self._correction_generation == self._context_generation)

    def _integrate_pending_key(self, vk, shift, caps, layout):
        if vk in DELIMITER_VKS:
            self._clear_buffers()
            return
        en_ch, he_ch = get_both_chars(vk, shift, caps)
        if en_ch:
            self.buffer_active += en_ch if layout == 'en' else he_ch
            self.buffer_shadow += he_ch if layout == 'en' else en_ch
        elif vk == VK_BACK:
            self.buffer_active = self.buffer_active[:-1]
            self.buffer_shadow = self.buffer_shadow[:-1]
        else:
            self._clear_buffers()


    # -------------------------------------------------------------------------
    # SECTION 2: OS KEY ENTRY, POLLING & LIFECYCLE
    # -------------------------------------------------------------------------

    def _on_key(self, vk, is_down):
        """Handle one physical key event from the OS layer.

        Returns:
            True to BLOCK the key from reaching the target application.
        """
        fresh = True
        if self._running and osapi.CHECK_CONTEXT_ON_KEY:
            fresh = self._refresh_input_context() is not None
        with self._context_lock:
            return self._on_key_locked(vk, is_down, fresh)

    def _on_key_locked(self, vk, is_down, fresh=True):
        if vk in (VK_SHIFT, VK_LSHIFT, VK_RSHIFT):
            self._shift_pressed = is_down
        elif vk in (VK_CONTROL, VK_LCONTROL, VK_RCONTROL):
            self._ctrl_pressed = is_down
        elif vk in (VK_MENU, VK_LMENU, VK_RMENU):
            self._alt_pressed = is_down
        elif vk in (VK_LWIN, VK_RWIN):
            self._win_pressed = is_down
        elif is_down and not self._check_suspend_hotkey(vk):
            if not fresh or (self._running and self._input_context is None):
                return False
            return self._handle_keypress(vk)
        return False

    def _poll_foreground_window(self):
        """Worker thread to poll OS state (Layout/Blacklist) every 100ms.
        
        This moves slow Windows API calls OUT of the keyboard hook callback
        to maintain system responsiveness.
        """
        while self._running:
            try:
                # Keep tracking focus during a correction so old work can cancel.
                self._refresh_input_context()

                # 2. Update Blacklist and IDE Status
                exe = self.blacklist.get_foreground_exe()
                is_blacklisted = exe in self.blacklist.blacklisted
                if not is_blacklisted:
                    try:
                        is_blacklisted = osapi.is_password_field_active()
                    except Exception as e:
                        logger.debug('Error checking password field: %s', e)
                self._cached_blacklisted = is_blacklisted
                self._cached_is_ide_editor = self.blacklist.is_ide_editor(exe)

                # 3. Detect Foreground Window changes (Trigger CRE)
                # Window/control and layout changes are handled together above.

                # 4. Check for Idle Timeout (Trigger CRE)
                if self.enabled and self.sensitivity.check_idle_timeout(self.idle_timeout):
                    self._fire_cre('idle_timeout')
                    self.sensitivity.record_keystroke() # Prevent infinite reset loop


            except Exception:
                logger.exception('Error in foreground poll')

            time.sleep(0.1)

    def start(self):
        """Launch the OS hooks and the OS polling thread."""
        self._running = True

        self._refresh_input_context()
        logger.info('Initial layout: %s', self._cached_layout)
        logger.info('Per-window input preference: %s', osapi.get_per_window_layout_enabled())

        # The hot keyboard hook and the mouse listener. May raise on Linux/macOS;
        # on Windows, install failures are logged in the hook thread.
        osapi.start_hooks(self._on_key, lambda: self._fire_cre('mouse_click'))

        # Slow poller for OS/UI context
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
        osapi.stop_hooks()
        logger.info('All hooks stopped')
