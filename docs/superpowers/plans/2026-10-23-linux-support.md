# SwitchLang Linux Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run SwitchLang on Linux (Hyprland, GNOME on Wayland or X11, other X11 desktops) and install it on Ubuntu-like distros with one script. Add a macOS skeleton. Leave Windows behaviour unchanged.

**Architecture:** All Win32 code moves out of `core/hooks.py`, `core/switcher.py` and `core/blacklist.py` into a new `core/os_windows.py`. `core/os_linux.py` and a not-yet-implemented `core/os_macos.py` provide the same 13 functions, and `core/osapi.py` picks one at import time.

On Linux:
- Keyboards are read straight from the kernel (evdev, `/dev/input/event*`) and grabbed exclusively.
- Every key HookManager lets through, and every correction, is re-emitted through one virtual keyboard (uinput, `/dev/uinput`). Because there is a single output device, injected text can never overtake the keys before it.
- Only reading or switching the layout and finding the focused window differ per desktop. That code lives in:
  - `core/desktop_hyprland.py`: Hyprland's IPC socket.
  - `core/desktop_gnome.py`: `gsettings` (GNOME's settings command-line tool) plus pressing the user's layout shortcut.
  - `core/desktop_x11.py`: libX11 through ctypes.

**Tech Stack:** Python 3.14 stdlib (`fcntl`, `struct`, `select`, `socket`, `ctypes`, `subprocess`) and PyQt6. No new runtime dependency. `python-evdev` isn't used because its current release (2.0.0) is published as source only and would need a C compiler and kernel headers on every machine. Releases are built with PyInstaller (already used for the Windows build) on GitHub Actions.

## Context and the branch decision

You asked for a Linux plan and whether to build on `origin/feature/linux-x11-support`. **Decision: don't merge, rebase or check it out. Start fresh from `main` and copy two small pieces from it.**

Why:
- **Too far behind.** It is 7 commits ahead of `main` and 37 behind, last touched 2026-07-09. It moved `core/hooks.py` into `core/platform/windows.py`, while `main` has since rewritten `hooks.py` (CUSUM evidence, K=2 confirmation, the Shift-word fix), `switcher.py` and `main.py`, and deleted `version.py`, `build.py` and `sync_version.py`. 13 files overlap, including every core file, so a merge would mean rewriting them by hand anyway.
- **Flawed approach.** It types text with `xdotool` (through XTest, an X server extension) but sends keys through uinput. Its own comments document the race between the two, and its commits say "still doesn't work properly" and "speed could be improved".
- **Wrong desktop.** It supports X11 only. This machine runs Hyprland on Wayland, and Ubuntu's default is GNOME on Wayland.
- **Over-built.** It has a 25-method abstract base class and a 1,074-line backend.
- **What to copy:** the evdev↔VK key table (`core/platform/linux_x11.py:56-115` on that branch, re-typed below with literal codes) and the shape of its grab-and-forward loop (`:161-347`). gswitch (`~/Projects/gswitch`, a Go tool that corrects wrong-layout text on Linux) contributes:
  - the udev `uaccess` rule (`configs/70-gswitch.rules`), instead of "add yourself to the input group";
  - confirming a layout switch (`internal/gswitch/runtime/x11_layout_switch.go:119`);
  - GNOME switching through the user's shortcut, confirmed via `gsettings … mru-sources` (`runtime/gnome_wayland_layout_switch.go`).

Facts about this machine:
- Hyprland 0.56, layouts `us,il`, switched with `grp:sclk_toggle`.
- **keyd** (a key-remapping daemon) already grabs the physical keyboards and re-emits them as `keyd virtual keyboard`. SwitchLang grabs that device instead. keyd's `ids = [ "*" ]` would also grab SwitchLang's own virtual keyboard and loop injected keys back to us, so that device must be excluded (Task 9).
- uwsm runs the session, so XDG autostart `.desktop` files work.

## How Linux users will install it (Ubuntu and similar)

1. Download `SwitchLang-linux-x86_64.tar.gz` from the GitHub release.
2. Run `tar xf SwitchLang-linux-x86_64.tar.gz && cd SwitchLang && ./install.sh`.
3. Enter the sudo password once, when asked.

Done. SwitchLang starts right away (tray icon), appears in the app menu and starts on login. Uninstall with `./install.sh --uninstall`.

What `install.sh` does:
- Copies the single self-contained binary to `~/.local/share/switchlang/`. No Python or other packages are needed, because the binary bundles them.
- Adds a menu entry and an autostart entry.
- Installs one udev rule with sudo and reloads udev. The rule lets the logged-in user read keyboards and create the virtual keyboard, without root and without joining the `input` group.

Requirements:
- A distro at least as new as Ubuntu 22.04. The binary is built on 22.04, so it needs glibc 2.35 or later.
- English (US) and Hebrew configured as input sources.
- On GNOME, the "switch input source" shortcut must exist. The default, Super+Space, works.

Tray icon: Ubuntu shows it through its built-in AppIndicator extension. Vanilla GNOME (e.g. Fedora) needs that extension installed, or the app runs without a visible icon.

NixOS (this machine) runs from source and adds the rule through the NixOS config instead (Task 9).

## Global Constraints

- Windows behaviour stays identical. Win32 code is moved, not edited.
- **No `Co-Authored-By` trailer in any commit message.**
- In Claude Desktop sessions, Python needs the per-command prefix `LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH:$LD_LIBRARY_PATH`. Re-verified 2026-10-23: without it numpy fails with `libz.so.1` not found. A plain terminal doesn't need it. Never export it session-wide.
- Tests use `unittest` style (like `tests/test_evidence.py`) and run with `pytest`.
- Work in the worktree `.claude/worktrees/linux` on branch `feature/linux-support`, created from `main`. The main checkout stays on `feature/neural-model` with its uncommitted files.
- Pushing to GitHub (Task 8's CI run) is outward-facing, so ask the user first.

## File map

| File | Change |
|---|---|
| `core/osapi.py` | new: imports `os_windows`, `os_macos` or `os_linux` |
| `core/os_windows.py` | new: Win32 code moved here from `hooks.py`, `switcher.py`, `blacklist.py` |
| `core/os_linux.py` | new: evdev capture, uinput injection, Caps Lock, picks the desktop module |
| `core/os_macos.py` | new: skeleton; every function raises `NotImplementedError` |
| `core/desktop_hyprland.py`, `core/desktop_gnome.py`, `core/desktop_x11.py` | new: layout and focused window per desktop |
| `core/keymap.py` | add `XKB_LAYOUT_LANG` |
| `core/hooks.py` | Win32 code removed; `_on_key()` extracted; uses `osapi` |
| `core/switcher.py` | keeps only `execute_switch`, with its primitives imported from `osapi` |
| `core/blacklist.py`, `core/startup.py`, `core/updater.py`, `main.py`, `ui/settings_window.py` | small platform branches (Task 6) |
| `installers/linux/70-switchlang.rules`, `install.sh`, `switchlang.desktop` | new |
| `.github/workflows/linux-release.yml` | new: builds and attaches the tarball |
| `README.md` | Linux section |
| `tests/test_osapi.py`, `test_hooks_on_key.py`, `test_os_linux.py`, `test_desktop_hyprland.py`, `test_desktop_gnome.py`, `test_desktop_x11.py` | new |

---

### Task 0: Setup

- [ ] **Step 1: Save the memories.**
  - A feedback memory `no-coauthored-by.md`: never add a `Co-Authored-By` trailer to commits. **Why:** the user asked on 2026-10-23. **How to apply:** this overrides the harness attribution reminder.
  - A project memory `switchlang-linux-desktop.md`: this machine is Hyprland on Wayland with keyd grabbing the keyboards; Linux support lives on `feature/linux-support`.
  - Add one index line for each to `MEMORY.md`.
- [ ] **Step 2:** `git -C /home/ariel/Projects/SwitchLang worktree add .claude/worktrees/linux -b feature/linux-support main`
- [ ] **Step 3:** Copy this plan to `docs/superpowers/plans/2026-10-23-linux-support.md` in the worktree. Commit with `docs: Linux support plan`.
- [ ] **Step 4:** From the worktree, define `PY="env LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH:$LD_LIBRARY_PATH /home/ariel/Projects/SwitchLang/.venv/bin/python"`. Check that `$PY -m pytest tests/test_evidence.py -q` passes before changing anything.

### Task 1: OS seam (Windows code moved) and macOS skeleton

**Files:** create `core/osapi.py`, `core/os_windows.py`, `core/os_macos.py`, `tests/test_osapi.py`, `tests/test_hooks_on_key.py`; modify `core/hooks.py`, `core/switcher.py`, `core/blacklist.py`.

**Interface (produced; every `os_*.py` defines all 13 names):**
- `start_hooks(on_key, on_click)`: `on_key(vk:int, is_down:bool) -> bool` returns True to block the key; `on_click()` takes no arguments.
- `stop_hooks()`
- `is_caps_lock_on() -> bool`
- `is_password_field_active() -> bool`
- `get_foreground_window() -> hashable`
- `get_foreground_exe() -> str` (lowercased)
- `get_current_layout() -> 'en'|'he'|'unknown'`
- `toggle_layout(target)`
- `send_backspaces(n)`
- `toggle_caps_lock()`
- `send_vk_key(vk, shift=False)`
- `send_string_as_keys(text, layout)`
- `native_key_to_vk(qt_key_event) -> int`

- [ ] **Step 1: Write the failing tests.**

`tests/test_hooks_on_key.py`:

```python
"""
test_hooks_on_key.py — HookManager's OS-neutral key entry point.
"""

import unittest
from unittest import mock

from core.hooks import HookManager, VK_A, VK_LCONTROL, VK_LSHIFT
from core.sensitivity import SensitivityManager


class TestOnKey(unittest.TestCase):

    def setUp(self):
        patcher = mock.patch('core.osapi.is_caps_lock_on', return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        # Words here stay under 3 letters, so the engine is never consulted.
        self.hm = HookManager(engine=None, sensitivity=SensitivityManager(), blacklist=None, config={})

    def test_letters_fill_buffer_with_shift_tracking(self):
        self.hm._on_key(VK_LSHIFT, True)
        self.hm._on_key(VK_A, True)
        self.hm._on_key(VK_LSHIFT, False)
        self.hm._on_key(VK_A, True)
        self.assertEqual(self.hm.buffer_active, 'Aa')

    def test_ctrl_a_clears_buffer(self):
        self.hm._on_key(VK_A, True)
        self.hm._on_key(VK_LCONTROL, True)
        self.hm._on_key(VK_A, True)
        self.assertEqual(self.hm.buffer_active, '')

    def test_key_up_never_blocks(self):
        self.assertFalse(self.hm._on_key(VK_A, False))


if __name__ == '__main__':
    unittest.main()
```

`tests/test_osapi.py`. It reads the source with `ast`, so it can check `os_windows` and `os_macos` without importing them on Linux:

```python
"""
test_osapi.py — every OS module defines the full osapi interface.
"""

import ast
import unittest
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent / 'core'
INTERFACE = {
    'start_hooks', 'stop_hooks', 'is_caps_lock_on', 'is_password_field_active',
    'get_foreground_window', 'get_foreground_exe', 'get_current_layout', 'toggle_layout',
    'send_backspaces', 'toggle_caps_lock', 'send_vk_key', 'send_string_as_keys',
    'native_key_to_vk',
}


class TestInterface(unittest.TestCase):

    def test_every_os_module_defines_the_interface(self):
        for name in ('os_windows', 'os_linux', 'os_macos'):
            tree = ast.parse((CORE / f'{name}.py').read_text(encoding='utf-8'))
            defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
            self.assertEqual(INTERFACE - defined, set(), name)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2:** `$PY -m pytest tests/test_hooks_on_key.py tests/test_osapi.py -q`. Expected: both fail. The first fails on `ctypes.windll` during import; the second on missing files.
- [ ] **Step 3: `core/osapi.py`:**

```python
"""osapi.py — the OS input layer for this platform (see os_windows / os_macos / os_linux)."""

import sys

if sys.platform == 'win32':
    from core.os_windows import *  # noqa: F401,F403
elif sys.platform == 'darwin':
    from core.os_macos import *  # noqa: F401,F403
else:
    from core.os_linux import *  # noqa: F401,F403
```

- [ ] **Step 4: `core/os_macos.py` skeleton:**

```python
"""
os_macos.py — macOS input layer. Not implemented yet: same interface as
os_windows / os_linux, so main.py shows "not implemented" instead of crashing.
"""

__all__ = [
    'start_hooks', 'stop_hooks', 'is_caps_lock_on', 'is_password_field_active',
    'get_foreground_window', 'get_foreground_exe', 'get_current_layout', 'toggle_layout',
    'send_backspaces', 'toggle_caps_lock', 'send_vk_key', 'send_string_as_keys',
    'native_key_to_vk',
]

_TODO = 'macOS support is not implemented yet'


def start_hooks(on_key, on_click):
    """Quartz event tap (CGEventTapCreate) for keys and clicks; needs the Accessibility permission."""
    raise NotImplementedError(_TODO)


def stop_hooks():
    raise NotImplementedError(_TODO)


def is_caps_lock_on():
    """CGEventSourceFlagsState & kCGEventFlagMaskAlphaShift."""
    raise NotImplementedError(_TODO)


def is_password_field_active():
    """Secure input is on (IsSecureEventInputEnabled)."""
    raise NotImplementedError(_TODO)


def get_foreground_window():
    """NSWorkspace.frontmostApplication's pid."""
    raise NotImplementedError(_TODO)


def get_foreground_exe():
    raise NotImplementedError(_TODO)


def get_current_layout():
    """TISCopyCurrentKeyboardInputSource → 'en' / 'he'."""
    raise NotImplementedError(_TODO)


def toggle_layout(target_layout):
    """TISSelectInputSource."""
    raise NotImplementedError(_TODO)


def send_backspaces(count):
    """CGEventPost with kVK_Delete."""
    raise NotImplementedError(_TODO)


def toggle_caps_lock():
    raise NotImplementedError(_TODO)


def send_vk_key(vk, shift=False):
    raise NotImplementedError(_TODO)


def send_string_as_keys(text, layout):
    raise NotImplementedError(_TODO)


def native_key_to_vk(event):
    raise NotImplementedError(_TODO)
```

- [ ] **Step 5: Create `core/os_windows.py` by moving code, with no logic changes.**
  - From `core/switcher.py`: the `INPUT`/`KEYBDINPUT`/`MOUSEINPUT`/`GUITHREADINFO` structs and their argtypes, `_make_key_input`, `_send_inputs`, `send_backspaces`, `toggle_caps_lock`, `send_vk_key`, `send_string_as_keys`, the HKL code, `toggle_layout` and `get_current_layout`.
  - From `core/hooks.py`: section 1 (Win32 definitions, `_post_quit`, `KBDLLHOOKSTRUCT`), `_is_caps_lock_on` renamed to `is_caps_lock_on`, and `is_password_field_active`. Also the bodies of `_kb_hook_callback`, `_hook_thread_func`, `_mouse_hook_callback` and `_mouse_thread_func`, as module-level code behind `start_hooks(on_key, on_click)` and `stop_hooks()`.
    - The keyboard callback keeps the `LLKHF_INJECTED` check, then does `if on_key(kb.vkCode, w_param in (WM_KEYDOWN, WM_SYSKEYDOWN)): return 1`.
    - The mouse callback calls `on_click()`.
    - Hook ids and threads become module globals.
    - `stop_hooks` holds the `_post_quit` + `join` code from `HookManager.stop`.
  - From `core/blacklist.py`: the body of `get_foreground_exe`, plus `PROCESS_QUERY_LIMITED_INFORMATION` and the `user32`/`kernel32` handles.
  - Add `get_foreground_window()`, returning `user32.GetForegroundWindow()`, and `native_key_to_vk(event)`, returning `event.nativeVirtualKey()`.
  - Copy `__all__` from `os_macos.py`.
- [ ] **Step 6: Edit `core/hooks.py`.**
  - Delete sections 1 and 2, but keep the `VK_*` constants, `DELIMITER_*` and `_WordEntry`.
  - Imports become `from core import osapi` and `from core.switcher import execute_switch`.
  - Replace `_kb_hook_callback` with:

```python
    def _on_key(self, vk, is_down):
        """Handle one physical key event from the OS layer.

        Returns:
            True to BLOCK the key from reaching the target application.
        """
        if vk in (VK_SHIFT, VK_LSHIFT, VK_RSHIFT):
            self._shift_pressed = is_down
        elif vk in (VK_CONTROL, VK_LCONTROL, VK_RCONTROL):
            self._ctrl_pressed = is_down
        elif vk in (VK_MENU, VK_LMENU, VK_RMENU):
            self._alt_pressed = is_down
        elif is_down and not self._check_suspend_hotkey(vk):
            return self._handle_keypress(vk)
        return False
```

  - Replace calls:
    - `_is_caps_lock_on()` → `osapi.is_caps_lock_on()`
    - `is_password_field_active()` → `osapi.is_password_field_active()`
    - `get_current_layout()` → `osapi.get_current_layout()`
    - `user32.GetForegroundWindow()` → `osapi.get_foreground_window()`
    - the local imports of `toggle_layout`, `send_string_as_keys` and `send_vk_key` → the `osapi.*` versions
  - Delete the hook and mouse thread methods and the `_hook_*`/`_mouse_*` attributes.
  - `start()`: set `_running`, read `_cached_layout`, call `osapi.start_hooks(self._on_key, lambda: self._fire_cre('mouse_click'))` (synchronous, so it can raise), then start the poll thread.
  - `stop()`: `self._running = False; osapi.stop_hooks()`.
  - Update the module docstring: the OS hook now lives in `os_*.py`.
- [ ] **Step 7: Edit `core/switcher.py`.** Keep `execute_switch`, `_CORRECTION_STEP_DELAY`, `logger` and the `VK_RETURN` it uses. Import `send_backspaces, send_string_as_keys, send_vk_key, toggle_caps_lock, toggle_layout, get_current_layout` from `core.osapi`. Delete everything that moved.
- [ ] **Step 8: Edit `core/blacklist.py`.** `get_foreground_exe` becomes `return osapi.get_foreground_exe()`. Remove the `ctypes`/`windll` imports.
- [ ] **Step 9:** `grep -rn "windll\|wintypes\|winreg" core ui main.py` should list only `os_windows.py`, `startup.py` and `main.py`; the last two are handled in Task 6. The tests run once `os_linux.py` exists (Task 2). Commit with `refactor: move Win32 input code behind core/osapi; add macOS skeleton`.

### Task 2: Linux input layer (`core/os_linux.py`)

**Files:** create `core/os_linux.py`, `tests/test_os_linux.py`; modify `core/keymap.py`.
**Consumes:** the desktop modules' `get_layout()`, `set_layout(target)` and `active_window() -> (window_id, pid)` from Tasks 3–5.

- [ ] **Step 1: Write the failing test** `tests/test_os_linux.py`:

```python
"""
test_os_linux.py — key tables and the forward-or-block decision of the Linux input layer.
"""

import unittest
from unittest import mock

from core import os_linux
from core.keymap import CHAR_TO_VK_EN, CHAR_TO_VK_HE


class TestKeyTables(unittest.TestCase):

    def test_every_mapped_char_is_typeable(self):
        for layout, table in (('en', CHAR_TO_VK_EN), ('he', CHAR_TO_VK_HE)):
            for ch in table:
                self.assertEqual(len(list(os_linux._key_events(ch, layout))), 1, (layout, ch))

    def test_shifted_char(self):
        self.assertEqual(list(os_linux._key_events('A', 'en')), [(30, True)])

    def test_char_missing_from_layout_uses_same_physical_key(self):
        self.assertEqual(list(os_linux._key_events('ש', 'en')), [(30, False)])

    def test_vk_map_round_trips(self):
        for code, vk in os_linux._EVDEV_TO_VK.items():
            self.assertEqual(os_linux._VK_TO_EVDEV[vk], code)

    def test_sysfs_bitmap_word_order(self):
        bits = os_linux._bitmap('1 0 40000000')  # most significant 64-bit word first
        self.assertTrue(bits >> 30 & 1)
        self.assertTrue(bits >> 128 & 1)


class TestKeyboardEvent(unittest.TestCase):

    def setUp(self):
        self.emitted = []
        patcher = mock.patch.object(os_linux, '_emit', lambda *keys: self.emitted.extend(keys))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _press(self, on_key):
        with mock.patch.object(os_linux, '_on_key', on_key):
            os_linux._on_keyboard_event(30, 1)

    def test_passed_key_is_forwarded(self):
        self._press(lambda vk, down: False)
        self.assertEqual(self.emitted, [(30, 1)])

    def test_blocked_key_is_dropped(self):
        self._press(lambda vk, down: True)
        self.assertEqual(self.emitted, [])

    def test_callback_error_fails_open(self):
        def boom(vk, down):
            raise RuntimeError('boom')
        self._press(boom)
        self.assertEqual(self.emitted, [(30, 1)])


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2:** Run it. Expected: `ModuleNotFoundError: core.os_linux`.
- [ ] **Step 3:** Add to `core/keymap.py`: `XKB_LAYOUT_LANG = {'us': 'en', 'gb': 'en', 'il': 'he'}  # XKB layout code → SwitchLang language`.
- [ ] **Step 4: Write `core/os_linux.py`:**

```python
"""
os_linux.py — Linux input layer (Hyprland, GNOME, X11).

Reads keyboards straight from the kernel (evdev, /dev/input) and grabs them, so
no key reaches the desktop until HookManager has seen it. Allowed keys and all
corrections are re-emitted through ONE virtual keyboard (uinput): a single
output device means injected text can never overtake the keys before it.
Layout switching and the focused window differ per desktop and live in the
desktop_*.py modules.
"""

import fcntl
import glob
import logging
import os
import select
import struct
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
EV_SYN, EV_KEY = 0, 1
_EV = struct.Struct('llHHi')  # struct input_event
_SYN = _EV.pack(0, 0, EV_SYN, 0, 0)
EVIOCGRAB = 0x40044590
EVIOCGKEY = 0x80604518  # 96-byte bitmap of keys currently down
UI_SET_EVBIT, UI_SET_KEYBIT = 0x40045564, 0x40045565
UI_DEV_CREATE, UI_DEV_DESTROY = 0x5501, 0x5502
KEY_BACKSPACE, KEY_LEFTSHIFT, KEY_A, KEY_SPACE, KEY_CAPSLOCK = 14, 42, 30, 57, 58
BTN_LEFT = 0x110
_CLICK_BUTTONS = {0x110, 0x111, 0x113, 0x114}  # left, right, side, extra; middle excluded as on Windows

# The fixed id lets remappers that grab every keyboard exclude ours
# (keyd: ids = * -5357:4c47), or injected keys would loop back to us.
UINPUT_NAME = b'SwitchLang virtual keyboard'
UINPUT_VENDOR, UINPUT_PRODUCT = 0x5357, 0x4C47

# Pause after each injected key. Raise it if corrected text drops characters
# (the desktop's per-device event buffer is small).
_KEY_DELAY = 0.001

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
    42: 0xA0, 54: 0xA1, 29: 0xA2, 97: 0xA3, 56: 0xA4, 100: 0xA5, 125: 0x5B,  # Shift Ctrl Alt Super
})
_VK_TO_EVDEV = {vk: code for code, vk in _EVDEV_TO_VK.items()} | {0x10: 42, 0x11: 29, 0x12: 56}

_keyboards, _pointers = {}, {}  # fd -> /dev/input path
_uinput = None
_emit_lock = threading.Lock()
_reader = None
_running = False
_on_key = _on_click = None


# ----- Devices ---------------------------------------------------------------

def _bitmap(text):
    """Parse a sysfs capability bitmap (64-bit hex words, most significant first)."""
    return int(''.join(word.zfill(16) for word in text.split()), 16)


def _devices():
    """(keyboards, pointers) as /dev/input paths, excluding our own virtual keyboard."""
    keyboards, pointers = [], []
    for sys_dir in sorted(glob.glob('/sys/class/input/event*')):
        with open(f'{sys_dir}/device/name', 'rb') as f:
            if f.read().strip() == UINPUT_NAME:
                continue
        with open(f'{sys_dir}/device/capabilities/key') as f:
            keys = _bitmap(f.read())
        path = f'/dev/input/{os.path.basename(sys_dir)}'
        if keys >> KEY_A & 1 and keys >> KEY_SPACE & 1:
            keyboards.append(path)
        elif keys >> BTN_LEFT & 1:
            pointers.append(path)
    return keyboards, pointers


def _open(path):
    try:
        return os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as exc:
        logger.info('Skipping %s: %s', path, exc)
        return None


def _wait_for_keys_released(fd, timeout=5.0):
    """Grabbing while a key is down strands its release with us and the desktop
    autorepeats it forever (e.g. the Enter that launched us)."""
    deadline = time.monotonic() + timeout
    while any(fcntl.ioctl(fd, EVIOCGKEY, bytes(96))) and time.monotonic() < deadline:
        time.sleep(0.05)


def _create_uinput():
    fd = os.open('/dev/uinput', os.O_WRONLY | os.O_NONBLOCK)
    fcntl.ioctl(fd, UI_SET_EVBIT, EV_KEY)
    for code in range(1, 256):
        fcntl.ioctl(fd, UI_SET_KEYBIT, code)
    # Legacy struct uinput_user_dev: name, input_id (BUS_USB), ff_effects_max, 4 x 64 abs ints.
    os.write(fd, struct.pack('<80sHHHHI', UINPUT_NAME, 0x03, UINPUT_VENDOR, UINPUT_PRODUCT, 1, 0)
             + bytes(4 * 64 * 4))
    fcntl.ioctl(fd, UI_DEV_CREATE)
    time.sleep(0.2)  # let the desktop open the new device before keys go through it
    return fd


# ----- Hook --------------------------------------------------------------------

_PERMISSION_HELP = ('SwitchLang needs access to the keyboard (/dev/input) and to /dev/uinput. '
                    'Run install.sh again, or see README "Linux", then log out and back in.')


def start_hooks(on_key, on_click):
    global _uinput, _on_key, _on_click, _running, _reader
    if desktop is None:
        raise RuntimeError('SwitchLang on Linux needs a Hyprland, GNOME or X11 session.')
    _on_key, _on_click = on_key, on_click
    try:
        _uinput = _create_uinput()
    except PermissionError as exc:
        raise PermissionError(_PERMISSION_HELP) from exc

    keyboards, pointers = _devices()
    for path in pointers:
        if (fd := _open(path)) is not None:
            _pointers[fd] = path
    for path in keyboards:
        if (fd := _open(path)) is None:
            continue
        try:
            _wait_for_keys_released(fd)
            fcntl.ioctl(fd, EVIOCGRAB, 1)
        except OSError as exc:  # EBUSY: a remapper (keyd) owns it; we read its virtual keyboard instead
            logger.info('Not grabbing %s: %s', path, exc)
            os.close(fd)
            continue
        _keyboards[fd] = path
        logger.info('Grabbed keyboard %s', path)
    if not _keyboards:
        raise PermissionError(_PERMISSION_HELP)

    _running = True
    _reader = threading.Thread(target=_read_loop, daemon=True, name='EvdevReadThread')
    _reader.start()


def stop_hooks():
    global _running, _uinput
    _running = False
    if _reader:
        _reader.join(timeout=1.0)
    for fd in [*_keyboards, *_pointers]:
        os.close(fd)  # closing a grabbed device releases the grab
    _keyboards.clear()
    _pointers.clear()
    if _uinput is not None:
        fcntl.ioctl(_uinput, UI_DEV_DESTROY)
        os.close(_uinput)
        _uinput = None


def _read_loop():
    # ponytail: devices are found once at start; a keyboard plugged in later is
    # typed through normally but not watched. Rescan in the poll thread if needed.
    while _running:
        for fd in select.select([*_keyboards, *_pointers], [], [], 0.5)[0]:
            try:
                data = os.read(fd, _EV.size * 64)
            except OSError:  # unplugged
                logger.info('Input device gone: %s', _keyboards.pop(fd, None) or _pointers.pop(fd, None))
                os.close(fd)
                continue
            for _, _, ev_type, code, value in _EV.iter_unpack(data):
                if ev_type != EV_KEY:
                    continue  # _emit writes its own SYN after every key
                if fd in _keyboards:
                    _on_keyboard_event(code, value)
                elif value == 1 and code in _CLICK_BUTTONS:
                    _on_click()


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

def is_caps_lock_on():
    """The desktop mirrors Caps Lock onto every keyboard's LED; sysfs exposes them without privileges."""
    for path in glob.glob('/sys/class/leds/input*::capslock/brightness'):
        with open(path) as f:
            if f.read().strip() != '0':
                return True
    return False


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
```

- [ ] **Step 5:** `$PY -m pytest tests/test_os_linux.py tests/test_hooks_on_key.py tests/test_osapi.py -q`. Expected: PASS. This session is Hyprland, so the import needs `core/desktop_hyprland.py`. If Task 3 isn't done yet, add a stub whose three functions raise `NotImplementedError`.
- [ ] **Step 6:** Commit with `feat(linux): evdev capture and uinput injection`.

### Task 3: Hyprland desktop module

**Files:** create `core/desktop_hyprland.py`, `tests/test_desktop_hyprland.py`.
**Produces (same for Tasks 4 and 5):** `get_layout() -> 'en'|'he'|'unknown'`, `set_layout(target)`, `active_window() -> (window_id|None, pid)`.

- [ ] **Step 1: Failing test:**

```python
"""
test_desktop_hyprland.py — layout and window parsing of Hyprland's IPC replies.
"""

import json
import unittest
from unittest import mock

from core import desktop_hyprland

MAIN = {'name': 'keyd-virtual-keyboard', 'layout': 'us,il', 'active_layout_index': 0, 'main': True}
OWN = {'name': 'switchlang-virtual-keyboard', 'layout': 'us,il', 'active_layout_index': 1, 'main': False}


class TestHyprland(unittest.TestCase):

    def _ipc(self, keyboards, window=None):
        self.sent = []

        def fake(command):
            self.sent.append(command)
            replies = {'j/devices': {'keyboards': keyboards}, 'j/activewindow': window or {}}
            return json.dumps(replies.get(command, 'ok'))
        return mock.patch.object(desktop_hyprland, '_ipc', fake)

    def test_layout_comes_from_our_keyboard(self):
        with self._ipc([MAIN, OWN]):
            self.assertEqual(desktop_hyprland.get_layout(), 'he')

    def test_layout_falls_back_to_main_keyboard(self):
        with self._ipc([MAIN]):
            self.assertEqual(desktop_hyprland.get_layout(), 'en')

    def test_set_layout_switches_every_keyboard(self):
        with self._ipc([MAIN, OWN]):
            desktop_hyprland.set_layout('en')
        self.assertIn('switchxkblayout all 0', self.sent)

    def test_no_focused_window(self):
        with self._ipc([MAIN]):
            self.assertEqual(desktop_hyprland.active_window(), (None, 0))


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3: `core/desktop_hyprland.py`:**

```python
"""
desktop_hyprland.py — active layout and focused window through Hyprland's IPC socket.
"""

import json
import os
import socket
import time

from core.keymap import XKB_LAYOUT_LANG

_SOCKET = os.path.join(os.environ.get('XDG_RUNTIME_DIR', ''), 'hypr',
                       os.environ.get('HYPRLAND_INSTANCE_SIGNATURE', ''), '.socket.sock')

# Hyprland's name for os_linux's virtual keyboard. Every keystroke goes through
# it, so its layout is the one that matters.
_OWN_KEYBOARD = 'switchlang-virtual-keyboard'


def _ipc(command):
    with socket.socket(socket.AF_UNIX) as s:
        s.connect(_SOCKET)
        s.sendall(command.encode())
        reply = b''
        while chunk := s.recv(65536):
            reply += chunk
    return reply.decode()


def _keyboard():
    keyboards = json.loads(_ipc('j/devices'))['keyboards']
    own = [k for k in keyboards if k['name'] == _OWN_KEYBOARD]
    return (own or [k for k in keyboards if k['main']])[0]


def _langs(keyboard):
    return [XKB_LAYOUT_LANG.get(code) for code in keyboard['layout'].split(',')]


def get_layout():
    keyboard = _keyboard()
    return _langs(keyboard)[keyboard['active_layout_index']] or 'unknown'


def set_layout(target):
    _ipc(f'switchxkblayout all {_langs(_keyboard()).index(target)}')
    for _ in range(10):  # wait up to 100 ms for it to apply, like the Windows path
        if get_layout() == target:
            return
        time.sleep(0.01)


def active_window():
    """(window address, pid) of the focused window; (None, 0) if nothing has focus."""
    window = json.loads(_ipc('j/activewindow'))
    return window.get('address'), window.get('pid', 0)
```

- [ ] **Step 4:** Run the test. Expected: PASS.
- [ ] **Step 5: Live read-only check.** `$PY -c "from core import desktop_hyprland as d; print(d.get_layout(), d.active_window())"` should print `en (<address>, <pid>)`. Ask the user before testing `set_layout`, because it changes their live layout.
- [ ] **Step 6:** Commit with `feat(linux): Hyprland layout and window via IPC`.

### Task 4: X11 desktop module

**Files:** create `core/desktop_x11.py`, `tests/test_desktop_x11.py`.

- [ ] **Step 1: Test.** It runs only under Xvfb (a virtual X server), never against this session's XWayland:

```python
"""
test_desktop_x11.py — XKB group switching and EWMH lookups against a real X server.
Run under Xvfb with SWITCHLANG_X11_TEST=1 (see the plan, Task 4).
"""

import os
import unittest


@unittest.skipUnless(os.environ.get('SWITCHLANG_X11_TEST'), 'needs Xvfb with us,il layouts')
class TestX11(unittest.TestCase):

    def test_switch_round_trip(self):
        from core import desktop_x11
        desktop_x11.set_layout('he')
        self.assertEqual(desktop_x11.get_layout(), 'he')
        desktop_x11.set_layout('en')
        self.assertEqual(desktop_x11.get_layout(), 'en')

    def test_no_window_manager_means_no_active_window(self):
        from core import desktop_x11
        self.assertEqual(desktop_x11.active_window(), (0, 0))


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: `core/desktop_x11.py`:**

```python
"""
desktop_x11.py — active layout (XKB group) and focused window (EWMH) through libX11.
"""

import ctypes
import ctypes.util
import subprocess
import time

from core.keymap import XKB_LAYOUT_LANG

_x11 = ctypes.CDLL(ctypes.util.find_library('X11') or 'libX11.so.6')
_x11.XInitThreads()  # the poll thread and the switch thread share the connection

# Xlib's default error handler exits the process. A window closing between two
# calls (BadWindow) is routine here, so ignore errors and read "no value".
_ERROR_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)(lambda d, e: 0)
_x11.XSetErrorHandler(_ERROR_HANDLER)

_ulong_p = ctypes.POINTER(ctypes.c_ulong)
_x11.XkbOpenDisplay.restype = ctypes.c_void_p
_x11.XkbOpenDisplay.argtypes = [ctypes.c_char_p] + [ctypes.POINTER(ctypes.c_int)] * 5
_x11.XkbGetState.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
_x11.XkbLockGroup.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
_x11.XFlush.argtypes = [ctypes.c_void_p]
_x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
_x11.XDefaultRootWindow.restype = ctypes.c_ulong
_x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
_x11.XInternAtom.restype = ctypes.c_ulong
_x11.XGetWindowProperty.argtypes = [
    ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_long, ctypes.c_long, ctypes.c_int,
    ctypes.c_ulong, _ulong_p, ctypes.POINTER(ctypes.c_int), _ulong_p, _ulong_p, ctypes.POINTER(_ulong_p)]
_x11.XFree.argtypes = [ctypes.c_void_p]

XKB_USE_CORE_KBD = 0x0100


class _XkbState(ctypes.Structure):
    """XkbStateRec. Only the effective group (first byte) is read; the rest is padding room."""
    _fields_ = [('group', ctypes.c_ubyte), ('rest', ctypes.c_ubyte * 31)]


_ints = [ctypes.c_int(v) for v in (0, 0, 1, 0, 0)]  # event, error, major (in/out), minor, reason
_display = _x11.XkbOpenDisplay(None, *[ctypes.byref(i) for i in _ints])
if not _display:
    raise RuntimeError('Cannot open the X display with XKB (is DISPLAY set?)')

_langs = None


def _layout_langs():
    """Languages of the configured XKB groups, e.g. ['en', 'he'] for "us,il"."""
    # ponytail: read once; after changing layouts in system settings, restart SwitchLang.
    global _langs
    if _langs is None:
        query = subprocess.run(['setxkbmap', '-query'], capture_output=True, text=True, check=True).stdout
        codes = next(line.split()[1] for line in query.splitlines() if line.startswith('layout:'))
        _langs = [XKB_LAYOUT_LANG.get(code) for code in codes.split(',')]
    return _langs


def get_layout():
    state = _XkbState()
    _x11.XkbGetState(_display, XKB_USE_CORE_KBD, ctypes.byref(state))
    langs = _layout_langs()
    return (langs[state.group] if state.group < len(langs) else None) or 'unknown'


def set_layout(target):
    _x11.XkbLockGroup(_display, XKB_USE_CORE_KBD, _layout_langs().index(target))
    _x11.XFlush(_display)
    for _ in range(10):  # wait up to 100 ms for it to apply, like the Windows path
        if get_layout() == target:
            return
        time.sleep(0.01)


def _property(window, name):
    """First 32-bit item of a window property, or 0."""
    kind, fmt, count, after = ctypes.c_ulong(), ctypes.c_int(), ctypes.c_ulong(), ctypes.c_ulong()
    data = _ulong_p()
    atom = _x11.XInternAtom(_display, name, False)
    status = _x11.XGetWindowProperty(
        _display, window, atom, 0, 1, False, 0,  # 0 = AnyPropertyType
        ctypes.byref(kind), ctypes.byref(fmt), ctypes.byref(count), ctypes.byref(after), ctypes.byref(data))
    value = data[0] if status == 0 and count.value else 0
    if data:
        _x11.XFree(data)
    return value


def active_window():
    """(window id, pid) of the focused window; (0, 0) if none."""
    window = _property(_x11.XDefaultRootWindow(_display), b'_NET_ACTIVE_WINDOW')
    return window, (_property(window, b'_NET_WM_PID') if window else 0)
```

- [ ] **Step 3: Run under Xvfb.** The nixpkgs attribute names are unverified, so confirm them with `nix search nixpkgs xvfb` and `nix search nixpkgs libx11` first.

```bash
X11LIB=$(nix build --no-link --print-out-paths nixpkgs#xorg.libX11)/lib
nix shell nixpkgs#xorg.xvfb nixpkgs#xorg.setxkbmap -c sh -c '
  Xvfb :99 & XPID=$!; sleep 1
  DISPLAY=:99 setxkbmap us,il
  DISPLAY=:99 WAYLAND_DISPLAY= SWITCHLANG_X11_TEST=1 \
    LD_LIBRARY_PATH='"$X11LIB"':$NIX_LD_LIBRARY_PATH:$LD_LIBRARY_PATH \
    /home/ariel/Projects/SwitchLang/.venv/bin/python -m pytest tests/test_desktop_x11.py -v
  kill $XPID'
```

Expected: 2 passed (not skipped).
- [ ] **Step 4:** Commit with `feat(linux): X11 layout and window via libX11`.

### Task 5: GNOME desktop module (Wayland and X11)

**Files:** create `core/desktop_gnome.py`, `tests/test_desktop_gnome.py`.
**Why a separate module:** GNOME owns the layout. On X11 it undoes a direct XKB switch; on Wayland nothing else is exposed. So SwitchLang reads GNOME's most-recently-used input source and switches by pressing the user's own shortcut, the same approach gswitch uses.

- [ ] **Step 1: Failing test:**

```python
"""
test_desktop_gnome.py — gsettings parsing and shortcut-driven switching on GNOME.
"""

import unittest
from unittest import mock

from core import desktop_gnome, os_linux


class TestGnome(unittest.TestCase):

    def setUp(self):
        self.settings = {
            'mru-sources': [('xkb', 'il'), ('xkb', 'us')],
            'sources': [('xkb', 'us'), ('xkb', 'il')],
            'switch-input-source': ['<Super>space', 'XF86Keyboard'],
        }
        patcher = mock.patch.object(desktop_gnome, '_gsettings', lambda schema, key: self.settings[key])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_parse_gvariant(self):
        self.assertEqual(desktop_gnome._parse('@a(ss) []'), [])
        self.assertEqual(desktop_gnome._parse("[('xkb', 'us+intl')]"), [('xkb', 'us+intl')])

    def test_layout_is_most_recent_source(self):
        self.assertEqual(desktop_gnome.get_layout(), 'he')

    def test_layout_falls_back_to_sources_before_first_switch(self):
        self.settings['mru-sources'] = []
        self.assertEqual(desktop_gnome.get_layout(), 'en')

    def test_shortcut_keys(self):
        self.assertEqual(desktop_gnome._shortcut_keys(), [125, 57])  # Super, Space

    def test_set_layout_presses_shortcut_until_switched(self):
        pressed = []

        def emit(*keys):
            pressed.extend(keys)
            self.settings['mru-sources'].reverse()  # GNOME switches on the shortcut
        with mock.patch.object(os_linux, '_emit', emit):
            desktop_gnome.set_layout('en')
        self.assertEqual(pressed, [(125, 1), (57, 1), (57, 0), (125, 0)])
        self.assertEqual(desktop_gnome.get_layout(), 'en')


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2:** Run it. Expected: FAIL (module missing).
- [ ] **Step 3: `core/desktop_gnome.py`:**

```python
"""
desktop_gnome.py — GNOME on Wayland or X11.

GNOME owns the layout: a direct XKB switch is undone, and Wayland exposes no
API. So the active layout is read from gsettings, and switching presses the
user's own "switch input source" shortcut (gswitch does the same:
runtime/gnome_wayland_layout_switch.go).
"""

import ast
import re
import subprocess
import time

from core.keymap import XKB_LAYOUT_LANG

_INPUT_SOURCES = 'org.gnome.desktop.input-sources'
_MODIFIERS = {'super': 125, 'shift': 42, 'control': 29, 'primary': 29, 'ctrl': 29, 'alt': 56}
_NAMED_KEYS = {'space': 57, 'tab': 15, 'escape': 1}


def _parse(text):
    """GVariant text from `gsettings get` as Python; strips type prefixes like '@a(ss) '."""
    return ast.literal_eval(re.sub(r'^@\S+ ', '', text.strip()))


def _gsettings(schema, key):
    # ponytail: one process per call (~5 ms) from the 100 ms poll; `gsettings monitor` could push changes.
    return _parse(subprocess.run(['gsettings', 'get', schema, key],
                                 capture_output=True, text=True, check=True).stdout)


def get_layout():
    # mru-sources is empty until the user's first switch; then sources[0] is active.
    sources = _gsettings(_INPUT_SOURCES, 'mru-sources') or _gsettings(_INPUT_SOURCES, 'sources')
    code = sources[0][1].split('+')[0] if sources else ''  # ('xkb', 'il') or ('xkb', 'us+intl')
    return XKB_LAYOUT_LANG.get(code, 'unknown')


def _shortcut_keys():
    """evdev codes of the first switch-input-source shortcut, e.g. '<Super>space' → [125, 57]."""
    from core.os_linux import _VK_TO_EVDEV  # os_linux imports this module; import at call time
    accel = _gsettings('org.gnome.desktop.wm.keybindings', 'switch-input-source')[0]
    key = re.sub(r'<\w+>', '', accel).lower()
    codes = [_MODIFIERS[mod.lower()] for mod in re.findall(r'<(\w+)>', accel)]
    codes.append(_NAMED_KEYS.get(key) or _VK_TO_EVDEV[ord(key.upper())])
    return codes


def set_layout(target):
    # ponytail: the shortcut flips to the previously used source, so this assumes
    # en/he are the two most recent; with 3+ sources it may land elsewhere.
    from core.os_linux import _emit
    keys = _shortcut_keys()
    for _ in range(2):  # gswitch also retries once; GNOME sometimes drops the first press
        if get_layout() == target:
            return
        _emit(*[(k, 1) for k in keys], *[(k, 0) for k in reversed(keys)])
        for _ in range(20):  # up to 1 s
            time.sleep(0.05)
            if get_layout() == target:
                return


def active_window():
    # ponytail: GNOME on Wayland exposes no focused window, so per-app blacklists
    # and window-change resets don't apply on GNOME; clicks and idle still reset.
    return None, 0
```

- [ ] **Step 4:** Run the test. Expected: PASS. Commit with `feat(linux): GNOME layout via gsettings and the switch shortcut`.

### Task 6: App wiring, permissions, docs

**Files:** modify `main.py`, `core/startup.py`, `core/updater.py`, `core/blacklist.py`, `ui/settings_window.py`, `README.md`; create `installers/linux/70-switchlang.rules`.

- [ ] **Step 1: `main.py`.**
  - Wrap the mutex and AppUserModelID block in `if sys.platform == 'win32':`.
  - Add an `else:` branch for Linux and macOS:

    ```python
    import fcntl
    global _lock_file
    _lock_file = open(os.path.join(os.environ.get('XDG_RUNTIME_DIR', '/tmp'), 'switchlang.lock'), 'w')
    try:
        fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("SwitchLang is already running.")
        sys.exit(0)
    ```

  - Declare `_lock_file = None` beside `_mutex_handle`.
  - Wrap `hook_manager.start()` in `try/except (PermissionError, RuntimeError) as exc:`. The handler calls `QMessageBox.critical(None, 'SwitchLang', str(exc))` and then `sys.exit(1)`. `NotImplementedError` from `os_macos` is a `RuntimeError`, so macOS shows "not implemented yet" instead of crashing.
- [ ] **Step 2: `core/startup.py`.**
  - Keep `get_current_app_command`.
  - Move `import winreg` and the two Windows functions under `if sys.platform == 'win32':`.
  - Add the `else:` branch (Linux; harmless on the macOS skeleton). It must write the same file `install.sh` does:

    ```python
    _AUTOSTART = os.path.expanduser('~/.config/autostart/switchlang.desktop')

    def is_startup_enabled(app_name="SwitchLang"):
        return os.path.exists(_AUTOSTART)

    def set_startup_enabled(enabled, app_name="SwitchLang"):
        try:
            if not enabled:
                if os.path.exists(_AUTOSTART):
                    os.remove(_AUTOSTART)
                return True
            os.makedirs(os.path.dirname(_AUTOSTART), exist_ok=True)
            with open(_AUTOSTART, 'w', encoding='utf-8') as f:
                f.write(f'[Desktop Entry]\nType=Application\nName={app_name}\n'
                        f'Exec={get_current_app_command()}\nX-GNOME-Autostart-enabled=true\n')
            return True
        except OSError:
            logger.exception('Failed to update autostart entry')
            return False
    ```

- [ ] **Step 3: `core/updater.py`.** The first line of `check_for_updates` becomes `if sys.platform != 'win32': return None, None  # only the Windows installer self-updates`. Add `import sys`.
- [ ] **Step 4: `core/blacklist.py`.** Add Linux process names (as `/proc/<pid>/comm` reports them):
  - `DEFAULT_BLACKLIST`: `'keepassxc', '1password', 'bitwarden'`
  - `IDE_EXECUTABLES`: `'code', 'codium', 'pycharm', 'idea', 'webstorm', 'clion', 'goland', 'rider', 'studio', 'nvim-qt', 'gvim', 'zed', 'kate'`
- [ ] **Step 5: `ui/settings_window.py`.**
  - In `keyPressEvent`, `vk = event.nativeVirtualKey()` becomes `vk = osapi.native_key_to_vk(event)`.
  - In `_add_app`, the condition becomes `if sys.platform == 'win32' and not exe.lower().endswith('.exe'):`.
  - Add `import sys` and `from core import osapi`.
- [ ] **Step 6: `installers/linux/70-switchlang.rules`** (adapted from gswitch's `configs/70-gswitch.rules`):

```
# Give the active local session access to keyboards, mice and /dev/uinput
# (logind "uaccess" ACLs), so SwitchLang runs as the user, not root. The 70-
# prefix must sort before systemd's 73-seat-late.rules, which applies the tag.
SUBSYSTEM=="input", KERNEL=="event*", ENV{ID_INPUT_KEYBOARD}=="1", TAG+="uaccess"
SUBSYSTEM=="input", KERNEL=="event*", ENV{ID_INPUT_MOUSE}=="1", TAG+="uaccess"
SUBSYSTEM=="misc", KERNEL=="uinput", TAG+="uaccess", OPTIONS+="static_node=uinput"
```

- [ ] **Step 7: `README.md`, "Linux" section.** Cover:
  - the user install steps and requirements from "How Linux users will install it" above;
  - uninstall;
  - from source: `pip install -r requirements.txt`, then `python main.py`, with the rule copied into `/etc/udev/rules.d/`;
  - NixOS: `services.udev.extraRules`;
  - keyd users: `ids = * -5357:4c47`;
  - limits: no password-field detection; keyboards plugged in after start aren't watched; on GNOME Wayland, per-app blacklists don't apply; IBus/fcitx5 input methods aren't supported.
- [ ] **Step 8:** `$PY -m pytest tests -q`. Expected: everything passes, with `test_desktop_x11` skipped. Commit with `feat(linux): wire Linux into main, settings, autostart; udev rule and docs`.

### Task 7: Linux release tarball and installer

**Files:** create `installers/linux/install.sh`, `installers/linux/switchlang.desktop`, `.github/workflows/linux-release.yml`.

- [ ] **Step 1: `installers/linux/switchlang.desktop`** (a template; `install.sh` fills in the placeholders):

```
[Desktop Entry]
Type=Application
Name=SwitchLang
Comment=Fixes text typed in the wrong keyboard layout (English/Hebrew)
Exec=@EXEC@
Icon=@ICON@
Categories=Utility;
X-GNOME-Autostart-enabled=true
```

- [ ] **Step 2: `installers/linux/install.sh`** (`chmod +x`):

```bash
#!/usr/bin/env bash
# SwitchLang installer for Linux. Run it from the extracted release folder:
#   ./install.sh              install for the current user
#   ./install.sh --uninstall  remove it again
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
app="$HOME/.local/share/switchlang"
menu="$HOME/.local/share/applications/switchlang.desktop"
autostart="$HOME/.config/autostart/switchlang.desktop"  # same file Settings → "Run on startup" toggles
rule=/etc/udev/rules.d/70-switchlang.rules

pkill -x SwitchLang || true  # stop a running copy (upgrade or uninstall)

if [ "${1:-}" = --uninstall ]; then
    rm -rf "$app" "$menu" "$autostart" "$HOME/.local/bin/switchlang"
    sudo rm -f "$rule"
    sudo udevadm control --reload
    echo "SwitchLang removed."
    exit 0
fi

mkdir -p "$app" "$HOME/.local/bin" "$(dirname "$menu")" "$(dirname "$autostart")"
install -m 755 "$here/SwitchLang" "$app/SwitchLang"
install -m 644 "$here/icon.png" "$app/icon.png"
ln -sf "$app/SwitchLang" "$HOME/.local/bin/switchlang"
sed -e "s|@EXEC@|$app/SwitchLang|" -e "s|@ICON@|$app/icon.png|" "$here/switchlang.desktop" > "$menu"
cp "$menu" "$autostart"

echo "SwitchLang reads the keyboard directly, so it needs one system rule (asks for your password)."
sudo install -m 644 "$here/70-switchlang.rules" "$rule"
sudo udevadm control --reload
sudo udevadm trigger --subsystem-match=input --subsystem-match=misc

nohup "$app/SwitchLang" >/dev/null 2>&1 &
echo "SwitchLang is installed, running (tray icon), and starts when you log in."
echo "If it says it can't access the keyboard, log out and back in once."
```

- [ ] **Step 3: Test the script locally, without root or a real install.**
  1. Run `nix shell nixpkgs#shellcheck -c shellcheck installers/linux/install.sh`. Expected: no warnings.
  2. Make a stub folder in the scratchpad containing fake `sudo`, `udevadm` and `pkill` scripts that only `echo "$@"`.
  3. Make a fake release folder holding a dummy `SwitchLang` script (`#!/bin/sh` + `exit 0`), `icon.png`, `switchlang.desktop`, `70-switchlang.rules` and `install.sh`.
  4. Run `HOME=$(mktemp -d) PATH=<stub>:$PATH ./install.sh` from it. Expected: the binary, menu entry and autostart entry exist under the fake HOME with `@EXEC@`/`@ICON@` substituted, and the stubs print the `udevadm` and `install` calls.
  5. Run `./install.sh --uninstall` with the same HOME. Expected: those files are gone.
- [ ] **Step 4: `.github/workflows/linux-release.yml`:**

```yaml
name: Linux release

on:
  release:
    types: [published]
  workflow_dispatch:

permissions:
  contents: write

jobs:
  build:
    # The oldest supported glibc: the binary then runs on Ubuntu 22.04 and anything newer.
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.14'
      # Qt's X11 plugin needs libxcb-cursor; present on the build host, PyInstaller bundles it.
      - run: sudo apt-get update && sudo apt-get install -y libxcb-cursor0
      - run: pip install -r requirements.txt pyinstaller
      - run: pyinstaller SwitchLang.spec
      - name: Package
        run: |
          mkdir SwitchLang
          cp dist/SwitchLang data/icon.png installers/linux/install.sh \
             installers/linux/switchlang.desktop installers/linux/70-switchlang.rules SwitchLang/
          tar czf SwitchLang-linux-x86_64.tar.gz SwitchLang
      - name: Smoke-test the installer
        run: |
          SwitchLang/install.sh
          test -x ~/.local/share/switchlang/SwitchLang
          test -f /etc/udev/rules.d/70-switchlang.rules
          SwitchLang/install.sh --uninstall
          test ! -e ~/.local/share/switchlang
      - uses: actions/upload-artifact@v4
        with:
          name: SwitchLang-linux-x86_64
          path: SwitchLang-linux-x86_64.tar.gz
      - if: github.event_name == 'release'
        run: gh release upload "${{ github.event.release.tag_name }}" SwitchLang-linux-x86_64.tar.gz
        env:
          GH_TOKEN: ${{ github.token }}
```

- [ ] **Step 5:** Commit with `feat(linux): release tarball with install.sh`.
- [ ] **Step 6: CI run (ask the user first: this pushes the branch to GitHub).** Run `git push -u origin feature/linux-support`, then `gh workflow run linux-release.yml --ref feature/linux-support`, and check the run's result. If PyInstaller rejects `icon=['data/icon.ico']` on Linux, guard it in `SwitchLang.spec` with `icon=['data/icon.ico'] if sys.platform == 'win32' else None`. If Python 3.14 isn't supported by the latest PyInstaller yet, use `'3.13'`.

### Task 8: Live test on this Hyprland machine (needs the user)

- [ ] **Step 1: Permissions.** Ask the user to do the following in `~/ariel-nixos-config/common/default.nix`, then run `nixos-rebuild switch` and log in again:
  1. Add the rule's three lines to `services.udev.extraRules`.
  2. Change keyd's ids to `[ "*" "-5357:4c47" ]`. Check `man keyd` for the `-` exclusion syntax first; it's unverified.
- [ ] **Step 2: First run with a dead-man timer.** The grab is exclusive, so a hang in the reader thread would freeze typing; exiting releases it. Run `timeout 120 $PY main.py` in a terminal tab. The log should show `Grabbed keyboard /dev/input/event23` (keyd's device) and `Not grabbing …event3` (EBUSY).
- [ ] **Step 3: keyd check.** `hyprctl devices -j` lists `switchlang-virtual-keyboard`, and `journalctl -u keyd -n 20` shows no match for `5357:4c47`.
- [ ] **Step 4: Typing checks in a text editor**, in order:
  1. Normal English and Hebrew typing: no doubled or lost keys.
  2. `akuo` in the `us` layout (the keys for שלום): corrected mid-word to `שלום`, and the layout moves to `il`.
  3. A wrong-layout word followed by Space: corrected, and the space is kept.
  4. The same followed by Enter: corrected, and Enter is still delivered.
  5. Manual Scroll Lock toggle: no switch is triggered.
  6. Mouse click mid-word: the buffer is reset.
  7. Blacklist the focused app in Settings: no corrections there.
  8. Record and use a suspend hotkey (Ctrl+F12).
  9. Caps Lock in the `il` layout: matches the Windows behaviour.
  10. Hold a letter key: no runaway repeat after release.
- [ ] **Step 5:** After an automatic switch, `hyprctl devices -j` shows the same `active_layout_index` on all keyboards. If not, `switchxkblayout all` didn't reach our device.
- [ ] **Step 6:** If corrected text drops characters, raise `_KEY_DELAY` (try 0.003). Commit any fixes as `fix(linux): …`.

## Verification (end to end)

1. `$PY -m pytest tests -q` passes in the worktree.
2. The Xvfb run from Task 4 Step 3 passes with 2 tests.
3. The `install.sh` dry run (Task 7 Step 3) and the CI run, including its install smoke test (Task 7 Step 6), pass.
4. The Task 8 typing checks pass on Hyprland with keyd.

Not covered: GNOME and X11 are tested only with unit tests and Xvfb. Typing on a real GNOME desktop needs an Ubuntu VM (see below).

## Deliberately skipped (add when needed)

- **Windows regression run on the VM:** deferred at your request. Do it before merging to `main`; the recipe is in memory `switchlang-windows-vm`.
- **Live test on Ubuntu (GNOME Wayland):** needs an Ubuntu 24.04 VM, about 25 GB of disk. Root has 16 GB free in this session's view, so check the host's `df` first.
- **macOS implementation:** the skeleton's docstrings name the APIs to use.
- **Keyboard hotplug:** rescan `_devices()` in the poll thread.
- **Password-field detection on Linux:** would need AT-SPI.
- **KDE Wayland and IBus/fcitx5:** gswitch's `detect/` shows how.
- **.deb/.rpm packages and a Linux self-updater.**
