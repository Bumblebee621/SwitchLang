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

import fcntl
import logging
import os
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
        self._keyboard_device = None
        self._uinput_device = None
        self._hook_thread = None
        self._running = False
        self._on_key_down = None
        self._on_key_up = None
        self._mouse_listener = None
        self._lock_file = None
        self._lock_fd = None

        # Shift state tracking (needed because evdev gives us raw events)
        self._shift_pressed = False

        # Layout group index mapping: discovered at startup
        self._en_group = 0
        self._he_group = 1
        self._xkb_display = None  # python-xlib Display for layout queries

        _build_keycode_maps()

    # ----- Keyboard Hook -----

    def start_keyboard_hook(self, on_key_down, on_key_up):
        import evdev

        self._on_key_down = on_key_down
        self._on_key_up = on_key_up
        self._running = True

        # 1. Find the keyboard device
        self._keyboard_device = self._find_keyboard_device()
        if not self._keyboard_device:
            logger.error(
                'Could not find keyboard device. '
                'Is the user in the "input" group? '
                'Run: sudo usermod -aG input $USER (then log out and back in)'
            )
            raise PermissionError(
                'Cannot access keyboard device. '
                'Please add your user to the "input" group:\n'
                '  sudo usermod -aG input $USER\n'
                'Then log out and back in.'
            )

        # 2. Create virtual output device (mirrors physical keyboard)
        self._uinput_device = evdev.UInput.from_device(
            self._keyboard_device,
            name='SwitchLang-Virtual-Keyboard'
        )

        # 3. Grab exclusive access to the physical keyboard
        self._keyboard_device.grab()
        logger.info(
            'Grabbed keyboard: %s (%s)',
            self._keyboard_device.name, self._keyboard_device.path
        )

        # 4. Start the read loop in a daemon thread
        self._hook_thread = threading.Thread(
            target=self._evdev_read_loop,
            daemon=True,
            name='EvdevHookThread'
        )
        self._hook_thread.start()

    def stop_keyboard_hook(self):
        self._running = False

        if self._keyboard_device:
            try:
                self._keyboard_device.ungrab()
                logger.info('Released keyboard grab')
            except Exception:
                pass

        if self._uinput_device:
            try:
                self._uinput_device.close()
            except Exception:
                pass

        if self._hook_thread and self._hook_thread.is_alive():
            self._hook_thread.join(timeout=2.0)

        logger.info('Keyboard hook stopped')

    def _find_keyboard_device(self):
        """Auto-detect the primary keyboard from /dev/input/event*."""
        import evdev
        from evdev import ecodes as e

        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]

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
                return dev

        return None

    def _evdev_read_loop(self):
        """Main loop: read events from the physical keyboard, forward or block."""
        from evdev import ecodes as e
        import evdev

        try:
            for event in self._keyboard_device.read_loop():
                if not self._running:
                    break

                # Only process key events
                if event.type != e.EV_KEY:
                    # Forward non-key events (SYN, etc.) as-is
                    self._uinput_device.write_event(event)
                    self._uinput_device.syn()
                    continue

                keycode = event.code
                vk = _EVDEV_TO_VK.get(keycode)

                if event.value == 1:  # Key down
                    # Track shift state locally
                    if keycode in (e.KEY_LEFTSHIFT, e.KEY_RIGHTSHIFT):
                        self._shift_pressed = True

                    if vk is not None and self._on_key_down:
                        block = self._on_key_down(vk, self._shift_pressed)
                        if block:
                            continue  # Swallow the event

                    # Forward the event
                    self._uinput_device.write(e.EV_KEY, keycode, 1)
                    self._uinput_device.syn()

                elif event.value == 0:  # Key up
                    if keycode in (e.KEY_LEFTSHIFT, e.KEY_RIGHTSHIFT):
                        self._shift_pressed = False

                    if vk is not None and self._on_key_up:
                        self._on_key_up(vk)

                    # Always forward key-up (never block releases)
                    self._uinput_device.write(e.EV_KEY, keycode, 0)
                    self._uinput_device.syn()

                elif event.value == 2:  # Key repeat
                    # Check if this key would be blocked on down
                    if vk is not None and self._on_key_down:
                        block = self._on_key_down(vk, self._shift_pressed)
                        if block:
                            continue

                    self._uinput_device.write(e.EV_KEY, keycode, 2)
                    self._uinput_device.syn()

        except OSError as exc:
            if self._running:
                logger.error('Keyboard device read error: %s', exc)
        finally:
            logger.info('evdev read loop exited')

    # ----- Input Injection -----

    def send_backspaces(self, count):
        from evdev import ecodes as e

        for _ in range(count):
            self._uinput_device.write(e.EV_KEY, e.KEY_BACKSPACE, 1)
            self._uinput_device.syn()
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
            result = subprocess.run(
                ['setxkbmap', '-query'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode != 0:
                return 'unknown'

            # Parse the 'layout:' line, e.g. "layout:     us,il"
            for line in result.stdout.splitlines():
                if line.strip().startswith('layout:'):
                    layouts_str = line.split(':', 1)[1].strip()
                    layouts = [l.strip() for l in layouts_str.split(',')]
                    break
            else:
                return 'unknown'

            # Get the current group index from xdotool
            group_result = subprocess.run(
                ['xdotool', 'get-active-window', 'get-window-focus', '--shell'],
                capture_output=True, text=True, timeout=2
            )

            # Fallback: use xkblayout-state or xset
            # For now use a simpler approach via /tmp or xkb-switch
            group_idx = self._get_xkb_group_index()
            if group_idx is not None and group_idx < len(layouts):
                layout = layouts[group_idx]
                return self._layout_to_lang(layout)

            # Fallback: just return based on first layout
            if layouts:
                return self._layout_to_lang(layouts[0])

            return 'unknown'
        except Exception as exc:
            logger.debug('Error detecting layout: %s', exc)
            return 'unknown'

    def _get_xkb_group_index(self):
        """Get the current XKB group index using xdotool or xkb-switch."""
        try:
            # Try xkb-switch first (most reliable)
            result = subprocess.run(
                ['xkb-switch', '-p'],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                current = result.stdout.strip()
                return self._lang_name_to_group(current)
        except FileNotFoundError:
            pass

        try:
            # Fallback: parse xset output for LED mask
            result = subprocess.run(
                ['xset', '-q'],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if 'LED mask' in line:
                        # Group state is in bits 13-14 of the LED mask
                        match = re.search(r'LED mask:\s+(\w+)', line)
                        if match:
                            mask = int(match.group(1), 16)
                            group = (mask >> 13) & 0x3
                            return group
        except FileNotFoundError:
            pass

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

    def toggle_layout(self, target):
        """Switch keyboard layout using XkbLockGroup via xdotool or xkb-switch."""
        try:
            # Determine which group index corresponds to the target
            target_xkb = 'us' if target == 'en' else 'il'

            # Try xkb-switch first (cleanest)
            try:
                subprocess.run(
                    ['xkb-switch', '-s', target_xkb],
                    check=True, timeout=2
                )
                logger.debug('Layout switched to %s via xkb-switch', target)
            except FileNotFoundError:
                # Fallback: use setxkbmap group locking
                group_idx = self._lang_name_to_group(target_xkb)
                if group_idx is not None:
                    subprocess.run(
                        ['xdotool', 'key', f'--delay', '0',
                         f'super+space' if group_idx == 1 else 'super+space'],
                        timeout=2
                    )
                else:
                    logger.warning('Could not determine group for layout %s', target)

            # Wait for layout to actually change
            for _ in range(10):
                if self.get_current_layout() == target:
                    break
                time.sleep(0.01)

        except Exception as exc:
            logger.error('Error toggling layout: %s', exc)

    # ----- System Queries -----

    def get_foreground_process(self):
        """Get the process name of the focused window via _NET_ACTIVE_WINDOW."""
        try:
            # Get the active window ID
            result = subprocess.run(
                ['xdotool', 'getactivewindow', 'getwindowpid'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode != 0:
                return ''

            pid = result.stdout.strip()
            if not pid or not pid.isdigit():
                return ''

            # Read the process name from /proc
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
        """Check Caps Lock state via xset or evdev LED state."""
        try:
            result = subprocess.run(
                ['xset', '-q'],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if 'Caps Lock' in line:
                        return 'on' in line.lower().split('caps lock')[1].split()[0]
        except Exception:
            pass
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
