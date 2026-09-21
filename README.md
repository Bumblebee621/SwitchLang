# SwitchLang

A low-latency, real-time keyboard layout auto-switcher for Windows (English ↔ Hebrew).

SwitchLang runs in the system tray, intercepts keystrokes, and automatically detects when you are typing in the wrong layout. When a mismatch is detected, it erases the incorrect characters, switches the OS keyboard layout, and reinjects the corrected text seamlessly.

## Features

- **Automatic Layout Correction:** Detects accidental typing in the wrong layout and switches seamlessly without dropping keystrokes.
- **Ultra-Lightweight:** Runs quietly in the background using under 2 MB of RAM with instant startup and zero typing lag.
- **Smart Programming Mode:** Automatically recognizes coding environments (IDEs, code editors, terminals) and adapts language detection to avoid false switches on code and commands.
- **Smart Context Resumption:** Automatically resets when you switch windows, click the mouse, pause typing, or select all text (`Ctrl + A`), avoiding false triggers.
- **Suspension Hotkey & Visual HUD:** Pause or resume auto-switching instantly with a customizable hotkey, accompanied by an on-screen display overlay.
- **Application Blacklist:** Ignore specific apps and games where layout switching should never trigger.
- **Adjustable Sensitivity:** Tune detection sensitivity to fit your typing speed and style.
- **System Startup & Auto-Updater:** Toggle launch on Windows boot and check for updates directly from the tray settings.

## Requirements

- Windows 10 or Windows 11
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

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details. Third-party license notices are documented in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
