# SwitchLang

A low-latency, real-time keyboard layout auto-switcher for Windows (English ↔ Hebrew).

SwitchLang runs in the system tray, intercepts keystrokes, and automatically detects when you are typing in the wrong layout. When a mismatch is detected, it erases the incorrect characters, switches the OS keyboard layout, and reinjects the corrected text seamlessly.

## Features

- **Automatic Layout Correction:** Detects accidental typing in the wrong layout and corrects the text.
- **Ultra-Lightweight:** Runs quietly in the background using under 2 MB of RAM with instant startup and zero typing lag.
- **Smart Programming Mode:** Automatically recognizes coding environments (IDEs, code editors, terminals) and adapts language detection to avoid false switches on code and commands.
- **Smart Context Resumption:** Automatically resets when you switch windows, click the mouse, pause typing, or select all text (`Ctrl + A`), avoiding false triggers.
- **Suspension Hotkey & Visual HUD:** Pause or resume auto-switching instantly with a customizable hotkey, accompanied by an on-screen display overlay.
- **Application Blacklist:** Ignore specific apps and games where layout switching should never trigger.
- **Adjustable Sensitivity:** Tune detection sensitivity to fit your typing speed and style.
- **System Startup & Auto-Updater:** Toggle launch on Windows boot and check for updates directly from the tray settings.

## Per-window keyboard layouts

SwitchLang follows the operating system's active layout. It does not assign or
restore its own language preference for each window. On Windows, this works with
the "Let me use a different input method for each app window" preference. On
Hyprland and X11, it follows layout changes made by the desktop or a window-layout
manager. These paths have automated regression coverage; live Windows and X11
per-window testing is still pending.

Changing windows or focused controls clears the previous typing context. A
pending correction stops when its context changes. Keys already queued for the
old context are discarded rather than replayed into another window; keys pressed
after a detected change pass through normally. Focus checks and global keyboard
injection are separate OS operations, so text already emitted before a focus
change is detected cannot be recalled.

GNOME's per-window mode is **not yet supported**. Its saved recent-source list
does not reliably report the layout restored on window focus. When per-window
mode is enabled, or its preference cannot be read, SwitchLang pauses automatic
correction and shows a message in the tray menu and tooltip. Normal typing
continues. To use automatic correction on GNOME for now, select one shared input
source for all windows in GNOME Settings. Global-mode GNOME retains its existing
limitations: focused-window identity and per-app blacklists are unavailable.

## Requirements

- Windows 10 or Windows 11, or Linux (see [Linux](#linux))
- English (US) and Hebrew (Standard) keyboard layouts installed
- Python 3.10+ (only required if running from source)

## Installation

### Installer (Recommended)

Download and run the latest `SwitchLang_Setup.exe` from the [GitHub Releases](https://github.com/Bumblebee621/SwitchLang/releases) page.

### Running from Source

```bash
# 1. Clone the repository
git clone https://github.com/Bumblebee621/SwitchLang.git
cd SwitchLang

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run SwitchLang
python main.py
```

Access settings by right-clicking the SwitchLang tray icon.

## Linux

Supported desktops: Hyprland, GNOME (Wayland or X11), and other X11 desktops.

### Requirements

- A distro at least as new as Ubuntu 22.04 (glibc 2.35+).
- English (US) and Hebrew input sources configured.
- On GNOME, a "switch input source" shortcut. The default (Super+Space) works; Alt+Shift should also work (untested).
- A tray: Ubuntu shows it through its built-in AppIndicator extension. Vanilla GNOME needs that extension installed.

### Install

1. Download `SwitchLang-linux-x86_64.tar.gz` from the [GitHub Releases](https://github.com/Bumblebee621/SwitchLang/releases) page.
2. Run `tar xf SwitchLang-linux-x86_64.tar.gz && cd SwitchLang && ./install.sh`
3. Enter your sudo password once when asked.

SwitchLang starts right away (tray icon), appears in the app menu, and starts on login. `install.sh` copies a single self-contained binary to `~/.local/share/switchlang/` (no Python needed), adds the menu and autostart entries, installs the udev rule with sudo, and reloads udev.

On Hyprland without uwsm or dex, autostart entries are ignored: add `exec-once = ~/.local/bin/switchlang` to `hyprland.conf` instead.

To uninstall, run `./install.sh --uninstall`.

### Running from source

1. `pip install -r requirements.txt`
2. Copy `installers/linux/70-switchlang.rules` to `/etc/udev/rules.d/`, then run `sudo udevadm control --reload && sudo udevadm trigger`.
3. Log out and back in (the rule grants access to the active session).
4. `python main.py`

### NixOS

Put the rule in your configuration instead of copying the file. Use `services.udev.packages`, not `services.udev.extraRules`: extraRules becomes `99-local.rules`, which runs after systemd's `73-seat-late.rules` applies the `uaccess` tag, so the tag would have no effect.

```nix
services.udev.packages = [
  (pkgs.writeTextDir "lib/udev/rules.d/70-switchlang.rules" ''
    SUBSYSTEM=="input", KERNEL=="event*", ENV{ID_INPUT_KEYBOARD}=="1", TAG+="uaccess"
    SUBSYSTEM=="input", KERNEL=="event*", ENV{ID_INPUT_MOUSE}=="1", TAG+="uaccess"
    SUBSYSTEM=="input", KERNEL=="event*", ENV{ID_INPUT_TOUCHPAD}=="1", TAG+="uaccess"
    SUBSYSTEM=="input", KERNEL=="event*", ENV{ID_INPUT_TOUCHSCREEN}=="1", TAG+="uaccess"
    SUBSYSTEM=="input", KERNEL=="event*", ENV{ID_INPUT_TABLET}=="1", TAG+="uaccess"
    SUBSYSTEM=="misc", KERNEL=="uinput", TAG+="uaccess", OPTIONS+="static_node=uinput"
  '')
];
```

### keyd

If you use keyd, exclude SwitchLang's virtual keyboard from it by adding `-5357:4c47` under `[ids]`, after `*`, in your keyd config (NixOS: `ids = [ "*" "-5357:4c47" ];`).

### Security

The udev rule lets any program you run read the keyboards and inject keys through `/dev/uinput`; that is the cost of not running SwitchLang as root. Unlike Windows, where the lock screen and admin prompts are out of reach, on Linux SwitchLang also sees what you type on the lock screen and in polkit or sudo password prompts. Typed text stays in memory and is never sent anywhere; it reaches disk only if you turn on debug logging, which logs corrected words.

### Limits

- No password-field detection.
- Keyboards that connect later (plugging in, Bluetooth, resume from suspend) are picked up automatically. If the log says they won't be watched, restart SwitchLang after connecting one.
- On GNOME Wayland, per-app blacklists don't apply.
- IBus and fcitx5 input methods aren't supported.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details. Third-party license notices are documented in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
