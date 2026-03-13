# Caps Lock Behavior in SwitchLang

## 1. Introduction: The OS Quirk
Handling Caps Lock in bilingual applications involves overcoming a very specific and often frustrating OS-level quirk. On Windows, when the active layout is set to **Hebrew** and **Caps Lock is ON**, the operating system stops outputting Hebrew characters. Instead, it forces all alphabetical keys to output **uppercase English characters** (e.g., pressing `ל` outputs `K`).

Before the recent changes, SwitchLang struggled with this state. Because the OS was outputting English letters, the app would get confused between what the user was physically pressing and what was appearing on the screen.

## 2. The Solution: Intent-Based Tracking
To fix this without breaking normal typing flows, we completely overhauled how Caps Lock is handled. The new logic is built on three pillars:

1. **Physical Key Tracking (`core/keymap.py`):** SwitchLang now strictly tracks the *physical* key you pressed via `get_both_chars`. Even if Caps Lock is ON and the OS forces an English `K` onto the screen, the internal engine knows you physically pressed the `ל` key.
2. **Intent Evaluation (`core/hooks.py`):** Inside `_handle_keypress`, the evaluation engine compares what you *intended* to type (based on physical keys in `buffer_active`) against what actually appeared on screen (`buffer_shadow`). 
3. **Programmatic Caps Lock Toggling (`core/switcher.py`):** SwitchLang now has the ability to simulate a Caps Lock keypress via the new `toggle_caps_lock` function to turn it OFF automatically during a text correction phase (`execute_switch`).

## 3. Scenarios: What happens when you type?

Here is a step-by-step breakdown of how the app behaves in different Caps Lock scenarios, and where the logic lives in the codebase.

### Scenario A: Typing English with Hebrew Layout & Caps Lock ON (e.g., "HELLO")
- **The Setup:** Your layout is **Hebrew**, Caps Lock is **ON**. You press the keys for `H E L L O`.
- **What the OS does:** Because Caps Lock is ON, the OS correctly prints `HELLO` to the screen.
- **What the app sees:** In `_handle_keypress`, the app sees your intended Hebrew text is `יקךךם` (the physical keys) but the English translation is `HELLO`. 
- **The Engine's Decision:** The engine evaluates the text and realizes `HELLO` makes perfect sense in English, while `יקךךם` is nonsense. Therefore, it confirms your intent is English.
- **The Action (`core/hooks.py` -> `_trigger_switch`):** Because you intended English, and the OS is *already* outputting English (due to Caps Lock), the logic in `_trigger_switch` actively decides to abort the switch (returning `False`). It **DOES NOTHING**, allowing you to continue typing "HELLO" seamlessly.

### Scenario B: Typing Hebrew with Hebrew Layout & Caps Lock ON (e.g., "KT YUC")
- **The Setup:** Your layout is **Hebrew**, Caps Lock is **ON**. You meant to type `לא טוב`, but because of Caps Lock, you are pressing the physical keys while the OS prints `KT YUC`.
- **What the OS does:** Prints `KT YUC`.
- **What the app sees:** The app tracks the physical keys and knows your intended text is `לא טוב` (`buffer_active`). The shadow text is `KT YUC` (`buffer_shadow`).
- **The Engine's Decision:** The engine evaluates the text and confirms that `לא` is a highly common Hebrew word, meaning your current Hebrew layout is exactly what you want. Normally, it would say `should_switch = False` (No switch needed).
- **The Action (Caps Lock Correction):** 
  - In `_handle_keypress`, the app realizes a paradox: *You want Hebrew, you are in Hebrew, but Caps Lock is ruining it.* It sets `needs_caps_fix = True` and forces a call to `_trigger_switch`.
  - Inside `_trigger_switch`, it sets `fix_caps = True` and performs a clever swap: `buf_active, buf_shadow = buf_shadow, buf_active`. This makes the engine erase the incorrect English letters and inject the correct Hebrew ones.
  - The thread then calls `execute_switch` (`core/switcher.py`), which uses Backspace to erase `KT YUC`, calls `toggle_caps_lock()` to turn Caps Lock **OFF**, and injects `לא טוב`. All of this happens *without* toggling the OS keyboard layout.

### Scenario C: Shift + Caps Lock Interaction
- **The Setup:** Caps Lock is **ON**. You hold **Shift** and press `K`.
- **What the OS does:** Standard OS behavior dictates that Shift inverses Caps Lock. It will output a lowercase `k`.
- **The Action (`core/keymap.py` -> `vk_to_char` / `get_both_chars`):** The app evaluates `effective_shift = shifted ^ caps_lock`. It natively understands `Shift XOR Caps Lock` for alphabetical characters. It accurately tracks that you output a lowercase letter, ensuring its internal buffers never fall out of sync with what is on your screen.

## Summary
SwitchLang now gracefully handles the Hebrew Caps Lock quirk. By separating raw physical keypresses from OS-level quirks (`core/keymap.py`), intelligently filtering user intent (`core/hooks.py`), and programmatically managing the keyboard state (`core/switcher.py`), the app creates a seamless typing experience regardless of Caps Lock misfires.