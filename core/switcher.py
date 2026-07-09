"""
switcher.py — Concurrency-safe keyboard layout correction sequence.

Handles the critical section: erasing wrong text, toggling the OS
keyboard layout, and injecting corrected text.

All OS-specific I/O (backspaces, text injection, layout toggle) is
delegated to the PlatformBackend instance.
"""

import logging
import time

logger = logging.getLogger('switchlang.switcher')

# Platform-normalised key codes (must match hooks.py constants)
VK_RETURN = 0x0D
_CORRECTION_STEP_DELAY = 0.005  # seconds between correction sub-steps


def execute_switch(buffer_active, buffer_shadow, target_layout,
                   platform,
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
        platform: PlatformBackend instance for OS-specific I/O.
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

    # Pipeline Synchronization: Wait for the kernel uinput queue to drain into the X server.
    # This prevents the user-space xdotool (XTEST) from overtaking the final gibberish
    # keystrokes traversing the kernel and deleting the wrong characters.
    time.sleep(0.05)

    # -------------------------------------------------------------------------
    # CLEAR FORWARD SELECTION (Autocomplete mitigation)
    # -------------------------------------------------------------------------
    # If the user's typing triggered an inline autocomplete (like Chrome's address
    # bar), the autocompleted text is highlighted as a forward selection.
    # We must explicitly overwrite it before sending backspaces. We do this by
    # injecting a "dummy" character from the target layout (so it won't trigger
    # a new autocomplete), and then erasing it along with the rest of the word.
    dummy_char = 'ש' if target_layout == 'he' else 'q'
    platform.send_text(dummy_char, platform.get_current_layout())
    erase_len += 1

    platform.send_backspaces(erase_len)
    time.sleep(_CORRECTION_STEP_DELAY)

    if fix_caps:
        platform.toggle_caps_lock()
        time.sleep(_CORRECTION_STEP_DELAY)

    current_layout = platform.get_current_layout()
    if target_layout != current_layout:
        platform.toggle_layout(target_layout)

    platform.send_text(inject_text, target_layout)
    time.sleep(_CORRECTION_STEP_DELAY)

    # Send Enter as a real VK keypress so apps like Discord recognise it.
    if needs_vk_return:
        platform.send_key(VK_RETURN)
        time.sleep(_CORRECTION_STEP_DELAY)
