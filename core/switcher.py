"""
switcher.py — Concurrency-safe keyboard layout correction sequence.

Handles the critical section: erasing wrong text, toggling the OS
keyboard layout, injecting corrected text, and flushing any keys
typed during the correction phase.

All OS-specific I/O (backspaces, text injection, layout toggle) is
delegated to the PlatformBackend instance.
"""

import logging
import time

from core.keymap import vk_to_char

logger = logging.getLogger('switchlang.switcher')

# Platform-normalised key codes (must match hooks.py constants)
VK_RETURN = 0x0D
_CORRECTION_STEP_DELAY = 0.005  # seconds between correction sub-steps


def execute_switch(buffer_active, buffer_shadow,
                   pending_queue, set_correcting, target_layout,
                   platform,
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
        platform: PlatformBackend instance for OS-specific I/O.
        correction_block: Optional list of _WordEntry namedtuples for
                          retroactively corrected colliding/ambiguous words that
                          preceded the trigger word.
        trigger_delimiter: The delimiter char (e.g. ' ') that was passed
                           through to the app before the switch thread ran,
                           or None for mid-word triggers.
        fix_caps: If True, toggle Caps Lock off during the correction.
    """
    set_correcting(True)

    try:
        # Determine if the trigger delimiter needs a real VK event.
        # Apps like Discord/Chrome require a real VK_RETURN to send a message;
        # a Unicode '\n' injected via KEYEVENTF_UNICODE is silently ignored.
        needs_vk_return = (trigger_delimiter == '\n')

        if correction_block:
            # Erase: trigger word (minus trigger char if mid-word) +
            #        each ambiguous word and the delimiter after it.
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

        # Pipeline Synchronization: Wait for the kernel uinput queue to drain into the X server.
        # This prevents the user-space xdotool (XTEST) from overtaking the final gibberish
        # keystrokes traversing the kernel and deleting the wrong characters.
        time.sleep(0.05)

        platform.replace_text(erase_len, inject_text)
        time.sleep(_CORRECTION_STEP_DELAY)

        current_layout = platform.get_current_layout()
        if target_layout != current_layout:
            platform.toggle_layout(target_layout)

        if fix_caps:
            platform.toggle_caps_lock()
            time.sleep(_CORRECTION_STEP_DELAY)

        # Send Enter as a real VK keypress so apps like Discord recognise it.
        if needs_vk_return:
            platform.send_key(VK_RETURN)
            time.sleep(_CORRECTION_STEP_DELAY)

        # Inject pending queue
        text_to_inject = ""
        consumed_pending = []
        while pending_queue:
            item = pending_queue.popleft()
            consumed_pending.append(item)
            q_vk, q_shift, q_caps = item

            # Map the raw typed key to a character in the TARGET layout.
            ch = vk_to_char(q_vk, q_shift, layout=target_layout, caps_lock=q_caps)
            if ch:
                text_to_inject += ch
        if text_to_inject:
            platform.send_unicode_string(text_to_inject)
        return consumed_pending
    finally:
        set_correcting(False)
