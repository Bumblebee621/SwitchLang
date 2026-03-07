# SwitchLang

A low-latency, real-time keyboard layout auto-switcher for Windows (English <-> Hebrew).

SwitchLang runs silently in the system tray, intercepts keystrokes, and uses an N-gram probabilistic model to detect when you're typing in the wrong layout. When it detects gibberish, it automatically:
1. Erases the wrong characters.
2. Toggles your OS keyboard layout.
3. Injects the correct characters.

## Features
- **Zero-Latency Hook:** Uses low-level Windows APIs (`WH_KEYBOARD_LL`).
- **Probabilistic Engine:** Character-level trigram scoring to minimize false positives.
- **Dynamic Sensitivity:** Threshold adapts dynamically based on context breaks (window focus change, mouse clicks, idle timeout, manual layout toggles).
- **Concurrency-Safe:** Queues physical keystrokes during the layout switch.
- **Diagnostics:** Auto-rotating application logs and CSV decision tracking.
- **App Blacklisting:** Frictionless tray UI to ignore specific applications.

## Requirements
- Windows OS
- Python 3.10+
- English (US) and Hebrew (Standard) keyboard layouts installed.

## Setup
```bash
# 1. Provide virtual environment (optional but recommended)
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate initial JSON n-gram models
python scripts/build_trigrams.py

# 4. Start the application
python main.py
```
*(Note: `pythonw main.py` runs it completely headless without a console).*

## Architecture
See `Game Plan.txt` for the core system design, evaluation pipeline, and state machine rules.
