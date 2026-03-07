# SwitchLang

A low-latency, real-time keyboard layout auto-switcher for Windows (English <-> Hebrew).

SwitchLang runs silently in the system tray, intercepts keystrokes, and uses an N-gram probabilistic model to detect when you're typing in the wrong layout. When it detects gibberish (e.g., typing "akuo" instead of "שלום"), it automatically:
1. Erases the wrong characters.
2. Toggles your OS keyboard layout.
3. Injects the correct characters.

## Features
- **Zero-Latency Hook:** Uses low-level Windows APIs (`WH_KEYBOARD_LL`).
- **Probabilistic Engine:** Character-level trigram scoring to minimize false positives.
- **Dynamic Sensitivity (CRE):** Threshold adapts dynamically based on context breaks (window focus change, mouse clicks, idle timeout).
- **Blacklisting:** Frictionless UI to ignore specific foreground applications.
- **Concurrency-Safe:** Queues physical keystrokes during the synthetic correction injection to prevent race conditions.

## Requirements
- Windows OS
- Python 3.10+
- Both English (US) and Hebrew (Standard) keyboard layouts installed in Windows settings.

## Setup
```bash
pip install -r requirements.txt
python scripts/build_trigrams.py  # Generates initial JSON models
python main.py                    # Starts the tray app & background service
```

*(Note: `pythonw main.py` runs it completely headless without a console).*

## Architecture
See `Game Plan.txt` for the core system design, evaluation pipeline, and state machine rules.
