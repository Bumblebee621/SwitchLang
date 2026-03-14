# SwitchLang

A low-latency, real-time keyboard layout auto-switcher for Windows (English <-> Hebrew).

SwitchLang runs in the system tray, intercepts keystrokes, and uses an N-gram probabilistic model to detect when you are typing in the wrong layout. When a mismatch is detected, it automatically erases the incorrect characters, toggles the OS keyboard layout, and reinjects the correct characters.

## Features

- **High-Density Engine:** Uses a character-level quadgram model trained on 10 million lines of conversational data (OpenSubtitles) for accurate language detection.
- **Context Resumption Events (CRE):** Automatically resets internal buffers and history on specific events, such as pressing `Ctrl + A` or changing window focus, to ensure fresh detection for new words.
- **Suspension Hotkey:** Support for a user-configurable keybind to temporarily pause auto-switching functionality.
- **Dynamic Sensitivity:** Thresholds adapt based on idle time, focus changes, and manual layout overrides.
- **Blacklisting:** Support for ignoring specific applications via the settings menu.

## Architecture

SwitchLang is built as a high-performance, concurrent-safe background service. It utilizes a two-tier evaluation pipeline to balance real-time responsiveness with probabilistic accuracy.

### 1. The Sensory System (Hooks)
The application installs a global low-level keyboard hook (`WH_KEYBOARD_LL`) to intercept physical keystrokes before they are committed to the active window. It maintains a dual-buffer state:
- **Active Buffer:** The characters as they appear in the current OS layout.
- **Shadow Buffer:** The same keystrokes mapped to the alternate layout.

### 2. Evaluation Pipeline
The decision to switch layouts relies on a two-tier process:
- **Tier 1: O(1) Collision Check:** Triggered on word delimiters (Space, Enter). The engine checks a pre-computed hash set of "Shadow Collisions"—words that are valid in both languages (e.g., "tbh" in English vs "אבה" in Hebrew keys).
- **Tier 2: Quadgram Model:** A character-level probabilistic model scores the active vs. shadow buffers. If the difference in log-probability exceeds a dynamic sensitivity threshold ($\Delta$), a correction is triggered.

### 3. Context Resumption Events (CRE)
A core concept in SwitchLang is the **Context Resumption Event**. This represents the moment the user's brain re-engages with the keyboard after a break in cognitive or digital flow. During these events, the engine resets its internal state and moves to maximum sensitivity.
CRE triggers include:
- **Window Focus Change:** Alt-tabbing or clicking a different application.
- **Mouse Clicks:** Physical mouse clicks that signal a caret jump.
- **Idle Timeout:** A 15-second gap between keystrokes.
- **Explicit Triggers:** Pressing `Ctrl + A` or manual layout toggles.

### 4. Concurrency-Safe Execution
When a switch is triggered, SwitchLang enters a temporary lock state (`is_correcting`). It erases the incorrect text, toggles the OS layout, and reinjects the shadow buffer using Unicode events. Any physical keys typed by the user during this millisecond-range operation are queued and flushed in the correct order after the switch is complete.

## Requirements

- Windows 10/11
- Python 3.10+
- English (US) and Hebrew (Standard) keyboard layouts installed.

## Setup

Run the following commands in your terminal (Command Prompt or PowerShell):

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the language models
python scripts/build_quadgrams.py
python scripts/build_collisions.py

# 4. Start the application
python main.py
```

Settings can be accessed by right-clicking the SwitchLang icon in the system tray.

