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

### Nix and NixOS (native package)

The repository provides a source-built Nix package with Python, Qt, and the language models included. Supports Linux on x86_64 and aarch64; aarch64 builds are not yet tested. Enable `nix-command` and `flakes` in your Nix settings to use the commands below.

Try it without installing:

```bash
nix run github:Bumblebee621/SwitchLang
```

Install it for your user:

```bash
nix profile add github:Bumblebee621/SwitchLang#switchlang
switchlang
```

These GitHub commands become available after the Nix files are committed and pushed. To build from a checkout, run `nix build` and then `./result/bin/switchlang`. New files must be tracked by Git for Nix to include them. The package also supports `nix-build` through `default.nix`, using your own `<nixpkgs>`.

**NixOS:** use the supplied module to install SwitchLang and configure device access. Add this input to your system flake:

```nix
inputs.switchlang.url = "github:Bumblebee621/SwitchLang";
```

Include `switchlang` in your flake's `outputs` arguments, then add `switchlang.nixosModules.default` to your host's `nixpkgs.lib.nixosSystem.modules` list. In that host's configuration:

```nix
programs.switchlang = {
  enable = true;
  autostart = true; # optional; defaults to false
};
```

Rebuild with `sudo nixos-rebuild switch`, then log out and back in. The module installs the existing `70-switchlang.rules` through `services.udev.packages` (before systemd applies `uaccess`) and loads `uinput`. This grants the active local session access to input devices; see [Security](#security).

Autostart uses a systemd user service attached to `graphical-session.target`. GNOME handles this automatically. Hyprland sessions must start that target and import their display and Hyprland environment into the user service manager (uwsm handles session integration). For a manually managed Hyprland session, leave `autostart = false` and add `exec-once = switchlang` to `hyprland.conf`. When the NixOS module is enabled, the app's startup checkbox is managed by the system configuration. Remove any older `~/.config/autostart/switchlang.desktop` entry when migrating to the module to avoid duplicate launches.

**Nix on other Linux distributions:** the package handles application dependencies. An administrator still needs to install `installers/linux/70-switchlang.rules`, reload udev, and ensure the `uinput` kernel module is loaded, as described in [Running from source](#running-from-source-1). Then log out and back in. Do not run the tarball's `install.sh` for the Nix package. You can enable per-user autostart in SwitchLang settings.

For profile installations, use `nix profile list` to find the installed entry name, then run `nix profile upgrade <name>`, or update the `switchlang` input in your system flake and rebuild. Remove a profile installation with `nix profile remove <name>`; for NixOS, remove the `programs.switchlang` settings and module import, then rebuild. User settings remain in `$XDG_CONFIG_HOME/SwitchLang` (default `~/.config/SwitchLang`).

For development and CI, `nix flake check` builds the package, runs the unit tests, and smoke-tests the installed Qt launcher offscreen. Keyboard access and real desktop switching still require testing in a graphical session with the udev rules active.

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
