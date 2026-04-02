"""
linux_x11.py — Linux / X11 platform backend.

Implements PlatformBackend using:
- evdev + uinput for keyboard hooking and injection (kernel-level,
  works on both X11 and future Wayland)
- python-xlib for layout detection and switching (XkbGetState / XkbLockGroup)
- /proc filesystem for process detection
- pynput for mouse listener (X11 only; future Wayland would use evdev)
- XDG standards for config and autostart

Requires the user to be in the 'input' group for evdev access.
"""

import ctypes
import ctypes.util
import fcntl
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from core.platform.base import PlatformBackend

logger = logging.getLogger('switchlang.platform.linux_x11')

# Virtual Key codes (Windows VK_* convention, used as normalised keycodes)
VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12    # Alt
VK_CAPITAL = 0x14
VK_SPACE = 0x20
VK_LSHIFT = 0xA0
VK_RSHIFT = 0xA1
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
VK_LMENU = 0xA4
VK_RMENU = 0xA5

# =============================================================================
# EVDEV KEYCODE <-> WINDOWS VK TRANSLATION TABLES
# =============================================================================

# These will be populated on first use (lazy import of evdev)
_EVDEV_TO_VK = None
_VK_TO_EVDEV = None


def _build_keycode_maps():
    """Build translation tables between evdev keycodes and Windows VK codes."""
    global _EVDEV_TO_VK, _VK_TO_EVDEV
    if _EVDEV_TO_VK is not None:
        return

    from evdev import ecodes as e

    # Map evdev KEY_* -> Windows VK_*
    _EVDEV_TO_VK = {
        # Letters
        e.KEY_A: 0x41, e.KEY_B: 0x42, e.KEY_C: 0x43, e.KEY_D: 0x44,
        e.KEY_E: 0x45, e.KEY_F: 0x46, e.KEY_G: 0x47, e.KEY_H: 0x48,
        e.KEY_I: 0x49, e.KEY_J: 0x4A, e.KEY_K: 0x4B, e.KEY_L: 0x4C,
        e.KEY_M: 0x4D, e.KEY_N: 0x4E, e.KEY_O: 0x4F, e.KEY_P: 0x50,
        e.KEY_Q: 0x51, e.KEY_R: 0x52, e.KEY_S: 0x53, e.KEY_T: 0x54,
        e.KEY_U: 0x55, e.KEY_V: 0x56, e.KEY_W: 0x57, e.KEY_X: 0x58,
        e.KEY_Y: 0x59, e.KEY_Z: 0x5A,

        # Numbers (top row)
        e.KEY_0: 0x30, e.KEY_1: 0x31, e.KEY_2: 0x32, e.KEY_3: 0x33,
        e.KEY_4: 0x34, e.KEY_5: 0x35, e.KEY_6: 0x36, e.KEY_7: 0x37,
        e.KEY_8: 0x38, e.KEY_9: 0x39,

        # Punctuation / symbols
        e.KEY_SEMICOLON: 0xBA,   # ;
        e.KEY_EQUAL: 0xBB,       # =
        e.KEY_COMMA: 0xBC,       # ,
        e.KEY_MINUS: 0xBD,       # -
        e.KEY_DOT: 0xBE,         # .
        e.KEY_SLASH: 0xBF,       # /
        e.KEY_GRAVE: 0xC0,       # `
        e.KEY_LEFTBRACE: 0xDB,   # [
        e.KEY_BACKSLASH: 0xDC,   # \
        e.KEY_RIGHTBRACE: 0xDD,  # ]
        e.KEY_APOSTROPHE: 0xDE,  # '

        # Special keys
        e.KEY_BACKSPACE: VK_BACK,
        e.KEY_TAB: VK_TAB,
        e.KEY_ENTER: VK_RETURN,
        e.KEY_SPACE: VK_SPACE,
        e.KEY_CAPSLOCK: VK_CAPITAL,

        # Modifiers
        e.KEY_LEFTSHIFT: VK_LSHIFT,
        e.KEY_RIGHTSHIFT: VK_RSHIFT,
        e.KEY_LEFTCTRL: VK_LCONTROL,
        e.KEY_RIGHTCTRL: VK_RCONTROL,
        e.KEY_LEFTALT: VK_LMENU,
        e.KEY_RIGHTALT: VK_RMENU,
    }

    # Build reverse map
    _VK_TO_EVDEV = {v: k for k, v in _EVDEV_TO_VK.items()}

    # Map generic VK_SHIFT/CONTROL/MENU to left variants for injection
    _VK_TO_EVDEV.setdefault(VK_SHIFT, e.KEY_LEFTSHIFT)
    _VK_TO_EVDEV.setdefault(VK_CONTROL, e.KEY_LEFTCTRL)
    _VK_TO_EVDEV.setdefault(VK_MENU, e.KEY_LEFTALT)


# =============================================================================
# LINUX X11 BACKEND
# =============================================================================


class LinuxX11Backend(PlatformBackend):
    """Linux platform backend using evdev + python-xlib."""

    def __init__(self):
        self._keyboard_devices = []
        self._uinput_device = None
        self._hook_threads = []
        self._running = False
        self._on_key_down = None
        self._on_key_up = None
        self._mouse_listener = None
        self._lock_file = None
        self._lock_fd = None

        self._event_queue = queue.Queue(maxsize=10000)
        self._worker_thread = None

        # Shift state tracking (needed because evdev gives us raw events)
        self._shift_pressed = False

        # Layout group index mapping: discovered at startup
        self._en_group = 0
        self._he_group = 1
        self._xkb_display = None  # python-xlib Display for layout queries

        _build_keycode_maps()

        # ctypes handle to libX11 and an open Display* for XkbGetState queries.
        # Lazily initialised once, reused on every 100ms poll (thread-safe: reads only).
        self._libX11 = None
        self._x11_display = None  # opaque Display* pointer
        self._x11_lock = threading.Lock()
        
        self._xlib_display = None
        self._cached_layouts = None

    # ----- Keyboard Hook -----

    def start_keyboard_hook(self, on_key_down, on_key_up):
        import evdev

        self._on_key_down = on_key_down
        self._on_key_up = on_key_up
        self._running = True

        # 1. Find all keyboard devices
        self._keyboard_devices = self._find_keyboard_devices()
        if not self._keyboard_devices:
            logger.error(
                'Could not find any keyboard devices. '
                'Is the user in the "input" group? '
                'Run: sudo usermod -aG input $USER (then log out and back in)'
            )
            raise PermissionError(
                'Cannot access keyboard device. '
                'Please add your user to the "input" group:\n'
                '  sudo usermod -aG input $USER\n'
                'Then log out and back in.'
            )

        # 2. Create virtual output device (mirrors capabilities of the first physical keyboard)
        self._uinput_device = evdev.UInput.from_device(
            self._keyboard_devices[0],
            name='SwitchLang-Virtual-Keyboard'
        )

        # 3. Grab exclusive access to all physical keyboards and start read loops
        for dev in self._keyboard_devices:
            try:
                dev.grab()
                logger.info('Grabbed keyboard: %s (%s)', dev.name, dev.path)
            except Exception as e:
                logger.warning('Could not grab keyboard %s: %s', dev.name, e)
                continue

            t = threading.Thread(
                target=self._evdev_read_loop,
                args=(dev,),
                daemon=True,
                name=f'EvdevHookThread-{os.path.basename(dev.path)}'
            )
            t.start()
            self._hook_threads.append(t)
        self._worker_thread = threading.Thread(
            target=self._evdev_worker_loop,
            daemon=True,
            name='EvdevWorkerThread'
        )
        self._worker_thread.start()

    def stop_keyboard_hook(self):
        self._running = False
        
        try:
            self._event_queue.put(None, timeout=1.0)
        except Exception:
            pass

        if hasattr(self, '_keyboard_devices') and self._keyboard_devices:
            for dev in self._keyboard_devices:
                try:
                    dev.ungrab()
                except Exception:
                    pass
            logger.info('Released keyboard grabs')

        if hasattr(self, '_uinput_device') and self._uinput_device:
            try:
                self._uinput_device.close()
            except Exception:
                pass

        if hasattr(self, '_hook_threads'):
            for t in self._hook_threads:
                if t.is_alive():
                    t.join(timeout=2.0)
        if hasattr(self, '_worker_thread') and self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

        logger.info('Keyboard hook stopped')

    def _find_keyboard_devices(self):
        """Auto-detect all primary keyboards from /dev/input/event*."""
        import evdev
        from evdev import ecodes as e

        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        found = []

        for dev in devices:
            caps = dev.capabilities(verbose=False)
            # Must have EV_KEY capability
            if e.EV_KEY not in caps:
                continue

            key_caps = caps[e.EV_KEY]
            # A real keyboard has letter keys (KEY_A=30 through KEY_Z=44+)
            has_letters = all(k in key_caps for k in [e.KEY_A, e.KEY_Z, e.KEY_SPACE, e.KEY_ENTER])
            if has_letters:
                # Skip virtual devices we created ourselves
                if 'switchlang' in dev.name.lower():
                    continue
                logger.info('Found keyboard: %s at %s', dev.name, dev.path)
                found.append(dev)

        return found

    def _evdev_read_loop(self, device):
        """Main loop: read events from the physical keyboard, forward or block."""
        from evdev import ecodes as e
        import evdev

        try:
            for event in device.read_loop():
                if not self._running:
                    break
                try:
                    self._event_queue.put(event, block=False)
                except queue.Full:
                    logger.critical('Evdev event queue is full! Dropping event from %s', device.name)
        except OSError as exc:
            if self._running:
                logger.error('Keyboard device read error: %s', exc)
        finally:
            logger.info('evdev read loop exited')

    def _evdev_worker_loop(self):
        """Worker thread: evaluate events sequentially without blocking the kernel."""
        from evdev import ecodes as e

        while self._running:
            try:
                event = self._event_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if event is None or not self._running:
                break

            # Only process key events
            if event.type != e.EV_KEY:
                self._uinput_device.write_event(event)
                continue

            keycode = event.code
            vk = _EVDEV_TO_VK.get(keycode)

            if event.value == 1:  # Key down
                if keycode in (e.KEY_LEFTSHIFT, e.KEY_RIGHTSHIFT):
                    self._shift_pressed = True

                if vk is not None and self._on_key_down:
                    try:
                        block = self._on_key_down(vk, self._shift_pressed)
                        if block:
                            continue  # Swallow the event
                    except Exception as exc:
                        logger.error('Crash in on_key_down: %s', exc, exc_info=True)

                self._uinput_device.write(e.EV_KEY, keycode, 1)

            elif event.value == 0:  # Key up
                if keycode in (e.KEY_LEFTSHIFT, e.KEY_RIGHTSHIFT):
                    self._shift_pressed = False

                if vk is not None and self._on_key_up:
                    try:
                        self._on_key_up(vk)
                    except Exception as exc:
                        logger.error('Crash in on_key_up: %s', exc, exc_info=True)

                self._uinput_device.write(e.EV_KEY, keycode, 0)

            elif event.value == 2:  # Key repeat
                if vk is not None and self._on_key_down:
                    try:
                        block = self._on_key_down(vk, self._shift_pressed)
                        if block:
                            continue
                    except Exception as exc:
                        logger.error('Crash in on_key_down (repeat): %s', exc, exc_info=True)

                self._uinput_device.write(e.EV_KEY, keycode, 2)

    # ----- Input Injection -----

    def send_backspaces(self, count):
        if count <= 0:
            return
            
        # Use xdotool for backspaces on X11 instead of uinput to avoid
        # race conditions. uinput goes through kernel -> evdev -> X11 queues,
        # whereas xdotool (XTest) bypasses the kernel entirely.
        # If we mix them, XTest text injections will overtake uinput backspaces,
        # causing the corrected text to be typed and then immediately deleted.
        try:
            subprocess.run(
                ['xdotool', 'key', '--delay', '0'] + ['BackSpace'] * count,
                check=True, timeout=2
            )
        except Exception as exc:
            logger.error('xdotool backspace error: %s', exc)
            # Fallback to uinput if xdotool fails
            from evdev import ecodes as e
            for _ in range(count):
                self._uinput_device.write(e.EV_KEY, e.KEY_BACKSPACE, 1)
                self._uinput_device.write(e.EV_KEY, e.KEY_BACKSPACE, 0)
            self._uinput_device.syn()

    def send_unicode_string(self, text):
        """Inject Unicode text using xdotool.

        xdotool's 'type' command handles Unicode characters correctly
        on X11 via XTest, regardless of the current keyboard layout.
        """
        if not text:
            return
        try:
            subprocess.run(
                ['xdotool', 'type', '--clearmodifiers', '--', text],
                check=True,
                timeout=5
            )
        except FileNotFoundError:
            logger.error(
                'xdotool not found. Install it with: sudo apt install xdotool'
            )
        except subprocess.TimeoutExpired:
            logger.error('xdotool timed out injecting text')
        except subprocess.CalledProcessError as exc:
            logger.error('xdotool error: %s', exc)

    def replace_text(self, erase_count, text):
        """Atomically replace text using a single xdotool command."""
        if erase_count <= 0 and not text:
            return

        cmd = ['xdotool']
        if erase_count > 0:
            cmd.extend(['key', '--delay', '0'] + ['BackSpace'] * erase_count)
        if text:
            cmd.extend(['type', '--clearmodifiers', '--', text])

        try:
            subprocess.run(cmd, check=True, timeout=5)
        except Exception as exc:
            logger.error('xdotool replace_text error: %s', exc)
            # Fallback to uinput for backspaces if xdotool failed entirely
            if erase_count > 0:
                from evdev import ecodes as e
                for _ in range(erase_count):
                    self._uinput_device.write(e.EV_KEY, e.KEY_BACKSPACE, 1)
                    self._uinput_device.write(e.EV_KEY, e.KEY_BACKSPACE, 0)
                self._uinput_device.syn()
            
            if text:
                logger.warning('xdotool fallback: dropped unicode text payload "%s"', text)

    def send_key(self, keycode):
        """Send a key press via uinput (used for Enter, etc.)."""
        from evdev import ecodes as e

        evdev_code = _VK_TO_EVDEV.get(keycode)
        if evdev_code is None:
            logger.warning('No evdev mapping for VK 0x%02X', keycode)
            return

        self._uinput_device.write(e.EV_KEY, evdev_code, 1)
        self._uinput_device.syn()
        self._uinput_device.write(e.EV_KEY, evdev_code, 0)
        self._uinput_device.syn()

    def toggle_caps_lock(self):
        from evdev import ecodes as e

        self._uinput_device.write(e.EV_KEY, e.KEY_CAPSLOCK, 1)
        self._uinput_device.syn()
        self._uinput_device.write(e.EV_KEY, e.KEY_CAPSLOCK, 0)
        self._uinput_device.syn()

    # ----- Layout Management -----

    def get_current_layout(self):
        """Detect the current XKB group and map it to 'en' or 'he'."""
        try:
            # Get the ordered list of configured layouts from setxkbmap (cached)
            if self._cached_layouts is None:
                self._cached_layouts = self._get_configured_layouts()
                
            layouts = self._cached_layouts
            if not layouts:
                return 'unknown'

            group_idx = self._get_xkb_group_index()
            # Self-healing cache recovery
            if group_idx is not None and group_idx >= len(layouts):
                logger.debug("Layout cache invalidated: group_idx %d >= len(layouts) %d", group_idx, len(layouts))
                self._cached_layouts = self._get_configured_layouts()
                layouts = self._cached_layouts
            if group_idx is not None and group_idx < len(layouts):
                return self._layout_to_lang(layouts[group_idx])

            # Fallback: assume first layout (English)
            return self._layout_to_lang(layouts[0])
        except Exception as exc:
            logger.debug('Error detecting layout: %s', exc)
            return 'unknown'

    def _get_configured_layouts(self):
        """Return the ordered list of XKB layout identifiers (e.g. ['us', 'il'])."""
        try:
            result = subprocess.run(
                ['setxkbmap', '-query'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode != 0:
                return []
            for line in result.stdout.splitlines():
                if line.strip().startswith('layout:'):
                    raw = line.split(':', 1)[1].strip()
                    return [l.strip() for l in raw.split(',')]
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # XKB group detection via ctypes + libX11  (no extra dependencies)
    # ------------------------------------------------------------------

    class _XkbStateRec(ctypes.Structure):
        """Mirror of XkbStateRec from <X11/XKBlib.h>."""
        _fields_ = [
            ('group',               ctypes.c_ubyte),
            ('locked_group',        ctypes.c_ubyte),
            ('base_group',          ctypes.c_ushort),
            ('latched_group',       ctypes.c_ushort),
            ('mods',                ctypes.c_ubyte),
            ('base_mods',           ctypes.c_ubyte),
            ('latched_mods',        ctypes.c_ubyte),
            ('locked_mods',         ctypes.c_ubyte),
            ('compat_state',        ctypes.c_ubyte),
            ('grab_mods',           ctypes.c_ubyte),
            ('compat_grab_mods',    ctypes.c_ubyte),
            ('lookup_mods',         ctypes.c_ubyte),
            ('compat_lookup_mods',  ctypes.c_ubyte),
            ('ptr_buttons',         ctypes.c_ushort),
        ]

    _XKB_USE_CORE_KBD = 0x0100

    def _ensure_libx11(self):
        """Lazily open a Display connection via XkbOpenDisplay.

        XkbOpenDisplay (unlike XOpenDisplay + XkbQueryExtension) allocates
        the dpy->xkb_info structure that XkbGetState requires internally.
        Without it, XkbGetState always returns False even if the extension
        is negotiated.

        Returns True if everything is ready, False if unavailable.
        """
        if self._libX11 is not None and self._x11_display is not None:
            return True
        try:
            lib = ctypes.CDLL('libX11.so.6')

            # XkbOpenDisplay allocates full XKB client state (dpy->xkb_info).
            # Signature: Display *XkbOpenDisplay(char *name, int *ev, int *err,
            #                                    int *major, int *minor, int *reason)
            lib.XkbOpenDisplay.restype  = ctypes.c_void_p
            lib.XkbOpenDisplay.argtypes = [
                ctypes.c_char_p,               # display name (NULL = $DISPLAY)
                ctypes.POINTER(ctypes.c_int),  # event_rtrn
                ctypes.POINTER(ctypes.c_int),  # error_rtrn
                ctypes.POINTER(ctypes.c_int),  # major_in_out (XkbMajorVersion)
                ctypes.POINTER(ctypes.c_int),  # minor_in_out (XkbMinorVersion)
                ctypes.POINTER(ctypes.c_int),  # reason_rtrn
            ]

            lib.XCloseDisplay.restype  = ctypes.c_int
            lib.XCloseDisplay.argtypes = [ctypes.c_void_p]

            lib.XFlush.restype = ctypes.c_int
            lib.XFlush.argtypes = [ctypes.c_void_p]

            lib.XkbGetState.restype  = ctypes.c_int   # Bool
            lib.XkbGetState.argtypes = [
                ctypes.c_void_p,                    # Display*
                ctypes.c_uint,                      # device_spec
                ctypes.POINTER(self._XkbStateRec),  # state_return
            ]

            lib.XkbLockGroup.restype = ctypes.c_int   # Bool
            lib.XkbLockGroup.argtypes = [
                ctypes.c_void_p,      # Display*
                ctypes.c_uint,        # device_spec
                ctypes.c_uint,        # group
            ]
            major  = ctypes.c_int(1)  # XkbMajorVersion
            minor  = ctypes.c_int(0)  # XkbMinorVersion
            event  = ctypes.c_int()
            error  = ctypes.c_int()
            reason = ctypes.c_int()

            display_ptr = lib.XkbOpenDisplay(
                None,
                ctypes.byref(event), ctypes.byref(error),
                ctypes.byref(major), ctypes.byref(minor),
                ctypes.byref(reason),
            )

            # XkbOD_Success = 0
            if not display_ptr or reason.value != 0:
                reason_names = {
                    1: 'BadLibraryVersion', 2: 'ConnectionRefused',
                    3: 'NonXkbServer',      4: 'BadServerVersion',
                }
                logger.warning(
                    'XkbOpenDisplay failed: reason=%s (%d)',
                    reason_names.get(reason.value, 'Unknown'), reason.value
                )
                return False

            self._libX11 = lib
            self._x11_display = display_ptr
            logger.debug('XkbOpenDisplay succeeded (event_base=%d).', event.value)
            return True
        except OSError as exc:
            logger.warning('Could not load libX11.so.6: %s', exc)
            return False

    def _get_xkb_group_index(self):
        """Return the current XKB group index (0-based integer).
        Uses ctypes XkbGetState via libX11 (primary, directly queries X server).
        """
        # --- Primary: ctypes + libX11.XkbGetState ---
        with self._x11_lock:
            try:
                if self._ensure_libx11():
                    state = self._XkbStateRec()
                    ok = self._libX11.XkbGetState(
                        self._x11_display,
                        self._XKB_USE_CORE_KBD,
                        ctypes.byref(state)
                    )
                    # XkbGetState returns Bool (1=True/success, 0=False/error) or Status=0
                    group = int(state.group)
                    if ok or (0 <= group <= 3):
                        return group
                    if self._x11_display:
                        self._libX11.XCloseDisplay(self._x11_display)
                    self._x11_display = None
                    self._libX11 = None
            except Exception as exc:
                logger.debug('ctypes XkbGetState error: %s', exc)
                if self._x11_display:
                    self._libX11.XCloseDisplay(self._x11_display)
                self._x11_display = None
                self._libX11 = None

        return None

        return None

    def _lang_name_to_group(self, name):
        """Map a layout name like 'us' or 'il' to a group index."""
        try:
            result = subprocess.run(
                ['setxkbmap', '-query'],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode != 0:
                return None

            for line in result.stdout.splitlines():
                if line.strip().startswith('layout:'):
                    layouts_str = line.split(':', 1)[1].strip()
                    layouts = [l.strip() for l in layouts_str.split(',')]
                    if name in layouts:
                        return layouts.index(name)
        except Exception:
            pass
        return None

    def _layout_to_lang(self, layout_name):
        """Map XKB layout identifiers to our 'en'/'he' convention."""
        layout_lower = layout_name.lower()
        if layout_lower in ('us', 'gb', 'en'):
            return 'en'
        elif layout_lower in ('il', 'he'):
            return 'he'
        return 'unknown'

    def _get_target_xkb_layout(self, target):
        """Find the actual layout string (e.g. 'us', 'gb', 'il') for the given target ('en' or 'he')."""
        try:
            result = subprocess.run(
                ['setxkbmap', '-query'],
                capture_output=True, text=True, timeout=1
            )
            for line in result.stdout.splitlines():
                if line.strip().startswith('layout:'):
                    layouts = [l.strip() for l in line.split(':', 1)[1].split(',')]
                    for l in layouts:
                        if self._layout_to_lang(l) == target:
                            return l
        except Exception:
            pass
        # Fallback to defaults
        return 'us' if target == 'en' else 'il'

    def toggle_layout(self, target):
        """Switch keyboard layout to 'target' ('en' or 'he').

        Strategy (order matters):
        1. XkbLockGroup — synchronous, driver-level, works on every X11 DE/WM.
           This is the *only* call that actually changes the X server's active
           group immediately.  It must be the primary path, not a fallback.
        2. GSettings notify (fire-and-forget) — tells GNOME/Cinnamon's session
           daemon what group we just switched to so their UI indicator stays in
           sync and, crucially, so the daemon does NOT detect the "unsolicited"
           XKB change and revert it by writing back the old group.
           We never gate on the exit code of these calls.
        """
        try:
            target_xkb = self._get_target_xkb_layout(target)
            group_idx = self._lang_name_to_group(target_xkb)

            if group_idx is None:
                logger.warning('Could not determine group index for layout %s', target)
                return

            # ----------------------------------------------------------------
            # Step 1 — XkbLockGroup (synchronous, ground-truth X11 group switch)
            # ----------------------------------------------------------------
            xkb_ok = False
            with self._x11_lock:
                try:
                    if self._ensure_libx11():
                        ok = self._libX11.XkbLockGroup(
                            self._x11_display,
                            self._XKB_USE_CORE_KBD,
                            group_idx,
                        )
                        if ok:
                            self._libX11.XFlush(self._x11_display)
                            xkb_ok = True
                            logger.debug(
                                'XkbLockGroup: switched to group %d (%s)', group_idx, target
                            )
                        else:
                            logger.warning('XkbLockGroup returned failure for group %d.', group_idx)
                            if self._x11_display:
                                self._libX11.XCloseDisplay(self._x11_display)
                            self._x11_display = None
                            self._libX11 = None
                except Exception as exc:
                    logger.debug('Error during XkbLockGroup: %s', exc)
                    if self._x11_display:
                        self._libX11.XCloseDisplay(self._x11_display)
                    self._x11_display = None
                    self._libX11 = None

            if not xkb_ok:
                logger.warning(
                    'XkbLockGroup unavailable; layout switch to %s may be unreliable.', target
                )

            # ----------------------------------------------------------------
            # Step 2 — GSettings notify (fire-and-forget, DE UI sync only)
            #
            # These writes tell the running GNOME/Cinnamon compositor which
            # group is now active so their on-screen indicator matches and so
            # they do not revert the XkbLockGroup call above.
            # We intentionally do NOT use check=True — a non-zero exit (e.g.
            # schema absent) is perfectly normal and should not mask xkb_ok.
            # ----------------------------------------------------------------
            for schema in (
                'org.cinnamon.desktop.input-sources',
                'org.gnome.desktop.input-sources',
            ):
                try:
                    subprocess.run(
                        ['gsettings', 'set', schema, 'current', str(group_idx)],
                        capture_output=True,
                        timeout=1,
                    )
                except Exception:
                    pass  # gsettings absent or schema not installed — normal on i3/openbox/etc.

            # ----------------------------------------------------------------
            # Step 3 — Verify (200 ms budget; XkbLockGroup is synchronous so
            # the first poll should already return the correct layout)
            # ----------------------------------------------------------------
            for _ in range(10):
                if self.get_current_layout() == target:
                    break
                time.sleep(0.02)
            else:
                logger.warning(
                    'Layout did not confirm as %s within 200 ms after toggle.', target
                )

        except Exception as exc:
            logger.error('Error toggling layout: %s', exc)

    # ----- System Queries -----

    def get_foreground_process(self):
        """Get the process name of the focused window via _NET_ACTIVE_WINDOW."""
        try:
            import Xlib.display
            import Xlib.X
            import Xlib.error

            if self._xlib_display is None:
                self._xlib_display = Xlib.display.Display()

            disp = self._xlib_display
            root = disp.screen().root
            net_active = disp.intern_atom('_NET_ACTIVE_WINDOW')
            
            act_win_prop = root.get_full_property(net_active, Xlib.X.AnyPropertyType)
            if not act_win_prop or not act_win_prop.value:
                return ''
                
            act_win_id = act_win_prop.value[0]
            if act_win_id == 0:
                return ''
                
            win = disp.create_resource_object('window', act_win_id)
            net_pid = disp.intern_atom('_NET_WM_PID')
            pid_prop = win.get_full_property(net_pid, Xlib.X.AnyPropertyType)
            
            if not pid_prop or not pid_prop.value:
                return ''
                
            pid = pid_prop.value[0]
            comm_path = f'/proc/{pid}/comm'
            if os.path.exists(comm_path):
                with open(comm_path, 'r') as f:
                    return f.read().strip().lower()
            return ''
            
        except ImportError:
            pass  # Fall back to xdotool
        except Exception as e:
            try:
                import Xlib.error
                if isinstance(e, Xlib.error.BadWindow):
                    return ''
                else:
                    if self._xlib_display is not None:
                        try:
                            self._xlib_display.close()
                        except Exception:
                            pass
                    self._xlib_display = None
                    return ''
            except ImportError:
                pass

        # xdotool Fallback
        try:
            result = subprocess.run(
                ['xdotool', 'getactivewindow', 'getwindowpid'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode != 0:
                return ''

            pid = result.stdout.strip()
            if not pid or not pid.isdigit():
                return ''

            comm_path = f'/proc/{pid}/comm'
            if os.path.exists(comm_path):
                with open(comm_path, 'r') as f:
                    return f.read().strip().lower()

            return ''
        except Exception as exc:
            logger.debug('Error getting foreground process: %s', exc)
            return ''

    def is_password_field_active(self):
        """Check for password fields using AT-SPI (graceful degradation)."""
        try:
            import pyatspi
            # Get the focused widget
            desktop = pyatspi.Registry.getDesktop(0)
            for app in desktop:
                try:
                    # Walk focused path
                    focus = app
                    # Check the focused object's role and state
                    if hasattr(focus, 'getRole'):
                        if focus.getRole() == pyatspi.ROLE_PASSWORD_TEXT:
                            return True
                    if hasattr(focus, 'getState'):
                        if focus.getState().contains(pyatspi.STATE_PROTECTED):
                            return True
                except Exception:
                    continue
        except ImportError:
            pass  # AT-SPI not available — graceful degradation
        except Exception as exc:
            logger.debug('AT-SPI error: %s', exc)

        return False

    def is_caps_lock_on(self):
        """Check Caps Lock state natively via libX11."""
        with self._x11_lock:
            try:
                if self._ensure_libx11():
                    state = self._XkbStateRec()
                    ok = self._libX11.XkbGetState(
                        self._x11_display,
                        self._XKB_USE_CORE_KBD,
                        ctypes.byref(state)
                    )
                    if ok:
                        # LockMask correctly correlates to bit 2
                        return bool(state.locked_mods & 2)

                    logger.debug('XkbGetState returned ok=%d; resetting display.', ok)
                    if self._x11_display:
                        self._libX11.XCloseDisplay(self._x11_display)
                    self._x11_display = None
                    self._libX11 = None
            except Exception as exc:
                logger.debug('ctypes XkbGetState error: %s', exc)
                if self._x11_display:
                    self._libX11.XCloseDisplay(self._x11_display)
                self._x11_display = None
                self._libX11 = None
        return False

    # ----- Mouse Listener -----

    def start_mouse_listener(self, on_click):
        from pynput import mouse as pynput_mouse
        self._mouse_listener = pynput_mouse.Listener(on_click=on_click)
        self._mouse_listener.daemon = True
        self._mouse_listener.start()

    def stop_mouse_listener(self):
        if self._mouse_listener:
            self._mouse_listener.stop()

    # ----- Application Lifecycle -----

    def set_single_instance_lock(self, app_name):
        """Use a file lock (fcntl.flock) for single-instance detection."""
        config_dir = self.get_config_dir()
        os.makedirs(config_dir, exist_ok=True)
        self._lock_file = os.path.join(config_dir, f'{app_name.lower()}.lock')

        try:
            self._lock_fd = open(self._lock_file, 'w')
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_fd.write(str(os.getpid()))
            self._lock_fd.flush()
            return True
        except (IOError, OSError):
            return False

    def release_single_instance_lock(self):
        if self._lock_fd:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                self._lock_fd.close()
            except Exception:
                pass
            self._lock_fd = None

        if self._lock_file and os.path.exists(self._lock_file):
            try:
                os.unlink(self._lock_file)
            except Exception:
                pass

    def get_config_dir(self):
        xdg_config = os.environ.get('XDG_CONFIG_HOME', os.path.expanduser('~/.config'))
        return os.path.join(xdg_config, 'switchlang')

    def is_startup_enabled(self, app_name='SwitchLang'):
        desktop_path = self._autostart_desktop_path(app_name)
        return os.path.exists(desktop_path)

    def set_startup_enabled(self, enabled, app_name='SwitchLang'):
        desktop_path = self._autostart_desktop_path(app_name)

        if enabled:
            os.makedirs(os.path.dirname(desktop_path), exist_ok=True)

            if getattr(sys, 'frozen', False):
                exec_cmd = sys.executable
            else:
                exec_cmd = f'{sys.executable} {os.path.abspath(sys.argv[0])}'

            content = (
                '[Desktop Entry]\n'
                f'Name={app_name}\n'
                'Type=Application\n'
                f'Exec={exec_cmd}\n'
                'Hidden=false\n'
                'NoDisplay=false\n'
                'X-GNOME-Autostart-enabled=true\n'
                f'Comment=Automatic keyboard layout switcher ({app_name})\n'
            )

            try:
                with open(desktop_path, 'w') as f:
                    f.write(content)
                logger.info('Created autostart desktop file: %s', desktop_path)
                return True
            except Exception as exc:
                logger.error('Error creating autostart file: %s', exc)
                return False
        else:
            try:
                if os.path.exists(desktop_path):
                    os.unlink(desktop_path)
                    logger.info('Removed autostart desktop file')
                return True
            except Exception as exc:
                logger.error('Error removing autostart file: %s', exc)
                return False

    def _autostart_desktop_path(self, app_name):
        xdg_config = os.environ.get('XDG_CONFIG_HOME', os.path.expanduser('~/.config'))
        return os.path.join(xdg_config, 'autostart', f'{app_name.lower()}.desktop')

    def set_app_id(self, app_id):
        # No-op on Linux — handled by .desktop files
        pass

    # ----- Keycode Translation -----

    def translate_keycode(self, native_keycode):
        _build_keycode_maps()
        return _EVDEV_TO_VK.get(native_keycode, native_keycode)

    def get_native_keycode(self, vk_code):
        _build_keycode_maps()
        return _VK_TO_EVDEV.get(vk_code, vk_code)
