"""
os_linux.py — Linux input layer (Hyprland, GNOME, X11).

Reads keyboards straight from the kernel (evdev, /dev/input) and grabs them, so
no key reaches the desktop until HookManager has seen it. Allowed keys and all
corrections are re-emitted through ONE virtual keyboard (uinput): a single
output device means injected text can never overtake the keys before it.
Layout switching and the focused window differ per desktop and live in the
desktop_*.py modules.
"""

import ctypes
import errno
import fcntl
import glob
import logging
import os
import select
import struct
import sys
import threading
import time

from core.keymap import CHAR_TO_VK_EN, CHAR_TO_VK_HE

if os.environ.get('HYPRLAND_INSTANCE_SIGNATURE'):
    from core import desktop_hyprland as desktop
elif 'GNOME' in os.environ.get('XDG_CURRENT_DESKTOP', ''):  # also Ubuntu's "ubuntu:GNOME"
    from core import desktop_gnome as desktop
elif os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    from core import desktop_x11 as desktop
else:
    desktop = None

__all__ = [
    'start_hooks', 'stop_hooks', 'is_caps_lock_on', 'is_password_field_active',
    'get_foreground_window', 'get_foreground_exe', 'get_current_layout', 'toggle_layout',
    'send_backspaces', 'toggle_caps_lock', 'send_vk_key', 'send_string_as_keys',
    'native_key_to_vk',
]

logger = logging.getLogger('switchlang.os')

# linux/input.h, linux/uinput.h (64-bit)
EV_SYN, EV_KEY, EV_LED = 0, 1, 0x11
_EV = struct.Struct('llHHi')  # struct input_event
_SYN = _EV.pack(0, 0, EV_SYN, 0, 0)
EVIOCGRAB = 0x40044590
EVIOCGKEY = 0x80604518  # 96-byte bitmap of keys currently down
UI_SET_EVBIT, UI_SET_KEYBIT, UI_SET_LEDBIT = 0x40045564, 0x40045565, 0x40045569
UI_DEV_CREATE, UI_DEV_DESTROY = 0x5501, 0x5502
LED_NUML, LED_CAPSL, LED_SCROLLL = 0, 1, 2
KEY_BACKSPACE, KEY_LEFTSHIFT, KEY_A, KEY_SPACE, KEY_CAPSLOCK = 14, 42, 30, 57, 58
BTN_LEFT, BTN_TOUCH = 0x110, 0x14A
# left, right, side, extra, touch; middle excluded as on Windows
_CLICK_BUTTONS = {0x110, 0x111, 0x113, 0x114, 0x14A}

# The fixed id lets remappers that grab every keyboard exclude ours
# (keyd: ids = * -5357:4c47), or injected keys would loop back to us.
UINPUT_NAME = b'SwitchLang virtual keyboard'
UINPUT_VENDOR, UINPUT_PRODUCT = 0x5357, 0x4C47

# Pause after each injected key. Raise it if corrected text drops characters
# (the desktop's per-device event buffer is small).
_KEY_DELAY = 0.001

IN_ATTRIB, IN_CREATE = 0x4, 0x100  # sys/inotify.h

# evdev key code → Windows VK code, so hooks.py keeps one set of constants.
_EVDEV_TO_VK = dict(zip(
    [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38,
     44, 45, 46, 47, 48, 49, 50],
    map(ord, 'QWERTYUIOPASDFGHJKLZXCVBNM')))
_EVDEV_TO_VK.update({code: 0x30 + d for d, code in enumerate([11, 2, 3, 4, 5, 6, 7, 8, 9, 10])})
_EVDEV_TO_VK.update({code: 0x70 + i for i, code in enumerate([59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 87, 88])})
_EVDEV_TO_VK.update({
    39: 0xBA, 13: 0xBB, 51: 0xBC, 12: 0xBD, 52: 0xBE, 53: 0xBF,        # ; = , - . /
    41: 0xC0, 26: 0xDB, 43: 0xDC, 27: 0xDD, 40: 0xDE,                  # ` [ \ ] '
    14: 0x08, 15: 0x09, 28: 0x0D, 57: 0x20, 58: 0x14, 1: 0x1B,         # Bksp Tab Enter Space Caps Esc
    119: 0x13, 70: 0x91, 104: 0x21, 109: 0x22, 107: 0x23, 102: 0x24,   # Pause ScrLk PgUp PgDn End Home
    105: 0x25, 103: 0x26, 106: 0x27, 108: 0x28, 110: 0x2D, 111: 0x2E,  # ← ↑ → ↓ Ins Del
    42: 0xA0, 54: 0xA1, 29: 0xA2, 97: 0xA3, 56: 0xA4, 100: 0xA5, 125: 0x5B, 126: 0x5C,  # Shift Ctrl Alt Super
})
_EVDEV_TO_VK.update({
    82: 0x60, 79: 0x61, 80: 0x62, 81: 0x63, 75: 0x64, 76: 0x65, 77: 0x66, 71: 0x67, 72: 0x68, 73: 0x69,  # KP0-KP9
    55: 0x6A, 78: 0x6B, 74: 0x6D, 83: 0x6E, 98: 0x6F,                  # KP* + - . /
    96: 0x0D,                                                          # KP Enter = VK_RETURN
})
# KP Enter (96) shares VK_RETURN with main Enter (28); skip it so main Enter wins the reverse map.
_VK_TO_EVDEV = {vk: code for code, vk in _EVDEV_TO_VK.items() if code != 96} | {0x10: 42, 0x11: 29, 0x12: 56}

_keyboards, _pointers = {}, {}  # fd -> /dev/input path
_uinput = None
_caps_lock = False  # mirrors the Caps Lock LED; updated from uinput LED events
_emit_lock = threading.Lock()
_reader = None
_inotify = None  # fd that turns readable when /dev/input changes (hotplug)
_running = False
_on_key = _on_click = None


# ----- Devices ---------------------------------------------------------------

def host_env():
    """Environment for running system tools. A PyInstaller one-file build points
    LD_LIBRARY_PATH at its bundled libraries (keeping the original in
    LD_LIBRARY_PATH_ORIG); child processes must get the system's back."""
    env = dict(os.environ)
    if 'LD_LIBRARY_PATH_ORIG' in env:
        env['LD_LIBRARY_PATH'] = env.pop('LD_LIBRARY_PATH_ORIG')
    elif getattr(sys, 'frozen', False):
        env.pop('LD_LIBRARY_PATH', None)
    return env


def _bitmap(text):
    """Parse a sysfs capability bitmap (64-bit hex words, most significant first)."""
    return int(''.join(word.zfill(16) for word in text.split()), 16)


def _sysfs(path):
    with open(path, 'rb') as f:
        return f.read().strip()


def _kind(keys, rel, abs_):
    """'keyboard', 'combo' (keyboard that also moves a pointer), 'pointer' or None,
    from the key/rel/abs sysfs capability bitmaps."""
    if keys >> KEY_A & 1 and keys >> KEY_SPACE & 1:
        return 'combo' if rel & 1 or abs_ & 1 else 'keyboard'  # REL_X / ABS_X
    if keys >> BTN_LEFT & 1 or keys >> BTN_TOUCH & 1:  # mouse, touchpad, touchscreen
        return 'pointer'
    return None


def _devices():
    """(keyboards, pointers) as /dev/input paths, excluding our own virtual keyboard."""
    keyboards, pointers = [], []
    for sys_dir in sorted(glob.glob('/sys/class/input/event*')):
        caps = f'{sys_dir}/device/capabilities'
        try:
            if _sysfs(f'{sys_dir}/device/name') == UINPUT_NAME:
                continue
            kind = _kind(*(_bitmap(_sysfs(f'{caps}/{c}').decode()) for c in ('key', 'rel', 'abs')))
        except OSError:  # unplugged while we looked
            continue
        path = f'/dev/input/{os.path.basename(sys_dir)}'
        if kind == 'keyboard':
            keyboards.append(path)
        elif kind:
            pointers.append(path)
            if kind == 'combo':  # grabbing it would freeze its pointer
                logger.info('%s is a keyboard with a pointer; its keys bypass SwitchLang', path)
    return keyboards, pointers


def _open(path, flags=os.O_RDONLY, level=logging.INFO):
    try:
        return os.open(path, flags | os.O_NONBLOCK)
    except OSError as exc:
        logger.log(level, 'Skipping %s: %s', path, exc)
        return None


def _wait_for_keys_released(fd, timeout=5.0):
    """Grabbing while a key is down strands its release with us and the desktop
    autorepeats it forever (e.g. the Enter that launched us)."""
    deadline = time.monotonic() + timeout
    while any(fcntl.ioctl(fd, EVIOCGKEY, bytes(96))) and time.monotonic() < deadline:
        time.sleep(0.05)


def _create_uinput():
    # O_RDWR: we also read the LED events the desktop writes back to this device.
    fd = os.open('/dev/uinput', os.O_RDWR | os.O_NONBLOCK)
    fcntl.ioctl(fd, UI_SET_EVBIT, EV_KEY)
    fcntl.ioctl(fd, UI_SET_EVBIT, EV_LED)
    for led in (LED_NUML, LED_CAPSL, LED_SCROLLL):
        fcntl.ioctl(fd, UI_SET_LEDBIT, led)
    for code in range(1, 256):
        fcntl.ioctl(fd, UI_SET_KEYBIT, code)
    # Legacy struct uinput_user_dev: name, input_id (BUS_USB), ff_effects_max, 4 x 64 abs ints.
    os.write(fd, struct.pack('<80sHHHHI', UINPUT_NAME, 0x03, UINPUT_VENDOR, UINPUT_PRODUCT, 1, 0)
             + bytes(4 * 64 * 4))
    fcntl.ioctl(fd, UI_DEV_CREATE)
    time.sleep(0.5)  # let the desktop (and any remapper) open the new device
    # A remapper that grabbed our device would feed our own keys back to us forever.
    own = [d for d in glob.glob('/sys/class/input/event*') if _sysfs(f'{d}/device/name') == UINPUT_NAME]
    for sys_dir in own:
        try:
            node = os.open(f'/dev/input/{os.path.basename(sys_dir)}', os.O_RDONLY)
            try:
                fcntl.ioctl(node, EVIOCGRAB, 1)
                fcntl.ioctl(node, EVIOCGRAB, 0)
            finally:
                os.close(node)
        except OSError as exc:
            if exc.errno != errno.EBUSY:
                logger.warning('Cannot check %s for remappers: %s', sys_dir, exc)
                continue
            fcntl.ioctl(fd, UI_DEV_DESTROY)
            os.close(fd)
            raise RuntimeError('Another program (keyd, kanata, input-remapper…) grabbed SwitchLang\'s '
                               'virtual keyboard; exclude device 5357:4c47 in its config (README, "keyd").') from exc
    if not own:
        logger.warning("SwitchLang's virtual keyboard has no event node; not checked for remappers")
    return fd


def _watch_dev_input():
    """inotify fd for new or newly accessible /dev/input nodes (IN_ATTRIB: udev
    grants the uaccess ACL after creating the node), or None without hotplug."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if fd < 0 or libc.inotify_add_watch(fd, b'/dev/input', IN_CREATE | IN_ATTRIB) < 0:
            err = ctypes.get_errno()
            if fd >= 0:
                os.close(fd)
            raise OSError(err, os.strerror(err))
        return fd
    except Exception as exc:
        logger.warning('Keyboards plugged in later will not be watched: %s', exc)
        return None


def _rescan(release_timeout=0.2):
    """Open the pointers and grab the keyboards that aren't open yet.
    Returns how many keyboards were skipped because another program grabbed them."""
    keyboards, pointers = _devices()
    known = {*_keyboards.values(), *_pointers.values()}
    for path in pointers:
        if path not in known and (fd := _open(path, level=logging.WARNING)) is not None:
            _pointers[fd] = path
    busy = 0
    for path in keyboards:
        # RDWR so the LED mirror writes in _on_led_event work
        if path in known or (fd := _open(path, os.O_RDWR)) is None:
            continue
        try:
            _wait_for_keys_released(fd, release_timeout)
            fcntl.ioctl(fd, EVIOCGRAB, 1)
        except OSError as exc:  # EBUSY: a remapper (keyd) owns it; we read its virtual keyboard instead
            logger.info('Not grabbing %s: %s', path, exc)
            os.close(fd)
            busy += 1
            continue
        _keyboards[fd] = path
        logger.info('Grabbed keyboard %s', path)
    return busy


# ----- Hook --------------------------------------------------------------------

_PERMISSION_HELP = ('SwitchLang needs access to the keyboard (/dev/input) and to /dev/uinput. '
                    'Run install.sh again, or see README "Linux", then log out and back in.')


def start_hooks(on_key, on_click):
    global _uinput, _on_key, _on_click, _running, _reader, _caps_lock, _inotify
    if desktop is None:
        raise RuntimeError('SwitchLang on Linux needs a Hyprland, GNOME or X11 session.')
    desktop.prepare()  # RuntimeError here reaches the startup dialog, before any word is erased
    _on_key, _on_click = on_key, on_click
    try:
        _uinput = _create_uinput()
    except PermissionError as exc:
        raise PermissionError(_PERMISSION_HELP) from exc
    _caps_lock = _read_caps_lock_sysfs()  # once, before grabbing; LED events keep it current after

    busy = _rescan(release_timeout=5.0)
    if not _keyboards:
        if busy:
            raise RuntimeError('Every keyboard is already grabbed by another program.')
        raise PermissionError(_PERMISSION_HELP)

    _inotify = _watch_dev_input()
    _running = True
    _reader = threading.Thread(target=_read_loop, daemon=True, name='EvdevReadThread')
    _reader.start()


def stop_hooks():
    global _running, _uinput, _inotify
    _running = False
    if _reader:
        _reader.join(timeout=1.0)
    if _inotify is not None:
        os.close(_inotify)
        _inotify = None
    for fd in [*_keyboards, *_pointers]:
        os.close(fd)  # closing a grabbed device releases the grab
    _keyboards.clear()
    _pointers.clear()
    if _uinput is not None:
        fcntl.ioctl(_uinput, UI_DEV_DESTROY)
        os.close(_uinput)
        _uinput = None


def _read_loop():
    try:
        while _running:
            # inotify last: unplugged devices are dropped before the rescan reopens their path.
            fds = [*_keyboards, *_pointers, _uinput] + ([_inotify] if _inotify is not None else [])
            try:
                ready = select.select(fds, [], [], 0.5)[0]
            except Exception:  # e.g. a closed fd; back off instead of spinning
                logger.exception('Error handling input event')
                time.sleep(0.5)
                continue
            for fd in ready:
                if fd == _inotify:
                    try:
                        os.read(fd, 65536)  # drain; what changed doesn't matter
                        _rescan()
                    except Exception:
                        logger.exception('Error rescanning input devices')
                    continue
                try:
                    data = os.read(fd, _EV.size * 64)
                except OSError:  # unplugged
                    if fd == _uinput:
                        continue
                    logger.info('Input device gone: %s', _keyboards.pop(fd, None) or _pointers.pop(fd, None))
                    os.close(fd)
                    continue
                try:
                    for _, _, ev_type, code, value in _EV.iter_unpack(data):
                        if fd == _uinput:
                            if ev_type == EV_LED:
                                _on_led_event(code, value)
                        elif ev_type != EV_KEY:
                            continue  # _emit writes its own SYN after every key
                        elif fd in _keyboards:
                            _on_keyboard_event(code, value)
                        elif value == 1 and code in _CLICK_BUTTONS:
                            _on_click()
                except Exception:
                    logger.exception('Error handling input event')  # keep the reader alive
    finally:
        # Reader gone (stop or crash): hand the keyboards back so keys reach the desktop.
        for fd in _keyboards:
            try:
                fcntl.ioctl(fd, EVIOCGRAB, 0)
            except OSError:
                pass


def _on_led_event(code, value):
    """The desktop writes LED state to our uinput device; keep Caps Lock current and
    pass the LED on to the grabbed keyboards so their lights follow."""
    global _caps_lock
    if code == LED_CAPSL:
        _caps_lock = bool(value)
    for fd in _keyboards:
        try:
            os.write(fd, _EV.pack(0, 0, EV_LED, code, value) + _SYN)
        except OSError as exc:
            logger.debug('LED mirror to %s failed: %s', _keyboards[fd], exc)


def _on_keyboard_event(code, value):
    """value: 1 = down, 2 = kernel autorepeat, 0 = up. Forward unless HookManager blocks it."""
    # ponytail: kernel autorepeat counts as a keypress, but the desktop repeats at
    # its own rate, so holding a letter can desync the buffer.
    vk = _EVDEV_TO_VK.get(code)
    if vk is not None:
        try:
            if _on_key(vk, value != 0):
                return
        except Exception:
            logger.exception('Error in keyboard callback')  # fail open: the key still reaches the app
    _emit((code, value))


# ----- Injection ---------------------------------------------------------------

def _emit(*keys):
    """Write (code, value) key events to the virtual keyboard, each followed by SYN."""
    with _emit_lock:
        os.write(_uinput, b''.join(_EV.pack(0, 0, EV_KEY, c, v) + _SYN for c, v in keys))


def _tap(code, shift=False):
    keys = [(code, 1), (code, 0)]
    if shift:
        keys = [(KEY_LEFTSHIFT, 1), *keys, (KEY_LEFTSHIFT, 0)]
    _emit(*keys)
    time.sleep(_KEY_DELAY)


def _key_events(text, layout):
    """Yield (evdev code, shift) pairs that type `text` while `layout` is active.

    A char missing from the layout's table is typed with the key that makes it
    in the other layout. That is all execute_switch's autocomplete dummy needs:
    any one character, which it erases straight after.
    """
    tables = (CHAR_TO_VK_EN, CHAR_TO_VK_HE) if layout == 'en' else (CHAR_TO_VK_HE, CHAR_TO_VK_EN)
    for ch in text:
        entry = tables[0].get(ch) or tables[1].get(ch)
        if entry is None:
            logger.warning('Cannot type %r on Linux; skipped', ch)
            continue
        vk, shift = entry
        yield _VK_TO_EVDEV[vk], shift


def send_string_as_keys(text, layout):
    for code, shift in _key_events(text, layout):
        _tap(code, shift)


def send_backspaces(count):
    for _ in range(count):
        _tap(KEY_BACKSPACE)


def toggle_caps_lock():
    _tap(KEY_CAPSLOCK)


def send_vk_key(vk, shift=False):
    code = _VK_TO_EVDEV.get(vk)
    if code is None:
        logger.warning('No evdev key for VK 0x%02X', vk)
        return
    _tap(code, shift)


# ----- State queries -------------------------------------------------------------

def _read_caps_lock_sysfs():
    """Caps Lock LED state from sysfs. Only read once, at start: grabbed keyboards
    drop the desktop's LED writes, so later changes come in as uinput LED events."""
    for path in glob.glob('/sys/class/leds/input*::capslock/brightness'):
        with open(path) as f:
            if f.read().strip() != '0':
                return True
    return False


def is_caps_lock_on():
    return _caps_lock


def is_password_field_active():
    # ponytail: Linux has no portable password-field signal; blacklist password
    # managers instead. AT-SPI (accessibility bus) could provide one.
    return False


def get_current_layout():
    try:
        return desktop.get_layout()
    except Exception:  # IPC hiccup or no desktop; the poll thread retries in 100 ms
        logger.debug('Layout query failed', exc_info=True)
        return 'unknown'


def toggle_layout(target_layout):
    desktop.set_layout(target_layout)


def get_foreground_window():
    return desktop.active_window()[0]


def get_foreground_exe():
    try:
        with open(f'/proc/{desktop.active_window()[1]}/comm') as f:
            return f.read().strip().lower()
    except OSError:
        return ''


def native_key_to_vk(event):
    """Qt's native scan code on Linux is the XKB keycode, i.e. evdev code + 8."""
    return _EVDEV_TO_VK.get(event.nativeScanCode() - 8, 0)
