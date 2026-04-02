"""
base.py — Abstract platform backend interface.

Defines the contract that every OS-specific backend must implement.
This keeps the core evaluation pipeline (hooks.py, switcher.py) completely
OS-agnostic: they call methods on a PlatformBackend instead of touching
ctypes, evdev, or X11 directly.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional


class PlatformBackend(ABC):
    """Interface every OS backend must implement.

    The core engine code (HookManager, execute_switch, etc.) operates
    entirely through this interface. Each platform provides its own
    implementation of keyboard hooking, input injection, layout management,
    and system queries.
    """

    # =========================================================================
    # KEYBOARD HOOK
    # =========================================================================

    @abstractmethod
    def start_keyboard_hook(
        self,
        on_key_down: Callable[[int, bool], bool],
        on_key_up: Callable[[int], None],
    ) -> None:
        """Install a global keyboard hook.

        The backend captures all physical key events and dispatches them
        to the provided callbacks.

        Args:
            on_key_down: Called on key-down events.
                - keycode: Platform-normalised virtual key code.
                - is_shifted: Whether Shift is currently held.
                Returns True to BLOCK the key, False to let it through.
            on_key_up: Called on key-up events.
                - keycode: Platform-normalised virtual key code.
        """

    @abstractmethod
    def stop_keyboard_hook(self) -> None:
        """Remove the keyboard hook and clean up resources."""

    # =========================================================================
    # INPUT INJECTION
    # =========================================================================

    @abstractmethod
    def send_backspaces(self, count: int) -> None:
        """Send N backspace key events to erase characters.

        Args:
            count: Number of backspaces to send.
        """

    @abstractmethod
    def send_unicode_string(self, text: str) -> None:
        """Inject a Unicode string into the focused application.

        Must work regardless of the current keyboard layout.

        Args:
            text: The string to inject.
        """

    @abstractmethod
    def replace_text(self, erase_count: int, text: str) -> None:
        """Batched operation: send backspaces, then inject text.
        
        This prevents race conditions on platforms like X11 by guaranteeing
        execution within the same client connection.
        
        Args:
            erase_count: Number of backspaces to send.
            text: The string to inject.
        """

    @abstractmethod
    def send_key(self, keycode: int) -> None:
        """Send a single key press (down + up) using the platform key code.

        Used for keys like Enter where apps expect a real key event
        rather than a Unicode character injection.

        Args:
            keycode: Platform-normalised virtual key code.
        """

    @abstractmethod
    def toggle_caps_lock(self) -> None:
        """Simulate a Caps Lock key press to toggle its state."""

    # =========================================================================
    # LAYOUT MANAGEMENT
    # =========================================================================

    @abstractmethod
    def get_current_layout(self) -> str:
        """Detect the currently active keyboard layout.

        Returns:
            'en' for English, 'he' for Hebrew, or 'unknown'.
        """

    @abstractmethod
    def toggle_layout(self, target: str) -> None:
        """Switch the OS keyboard layout to the specified target.

        Should block until the switch is confirmed or a timeout expires.

        Args:
            target: 'en' or 'he'.
        """

    # =========================================================================
    # SYSTEM QUERIES
    # =========================================================================

    @abstractmethod
    def get_foreground_process(self) -> str:
        """Get the executable/process name of the currently focused window.

        Returns:
            Process name (e.g. 'code.exe' on Windows, 'code' on Linux),
            or '' on failure.
        """

    @abstractmethod
    def is_password_field_active(self) -> bool:
        """Check if the user is typing in a password field.

        Returns:
            True if the currently focused control is a password input.
            Returns False if detection is unavailable (graceful degradation).
        """

    @abstractmethod
    def is_caps_lock_on(self) -> bool:
        """Check if Caps Lock is currently activated.

        Returns:
            True if Caps Lock is on.
        """

    # =========================================================================
    # MOUSE LISTENER
    # =========================================================================

    @abstractmethod
    def start_mouse_listener(self, on_click: Callable) -> None:
        """Start listening for global mouse click events.

        Args:
            on_click: Callback with signature (x, y, button, pressed).
        """

    @abstractmethod
    def stop_mouse_listener(self) -> None:
        """Stop the mouse click listener."""

    # =========================================================================
    # APPLICATION LIFECYCLE
    # =========================================================================

    @abstractmethod
    def set_single_instance_lock(self, app_name: str) -> bool:
        """Attempt to acquire a single-instance lock.

        Args:
            app_name: Application identifier.

        Returns:
            True if the lock was acquired (we are the first instance).
            False if another instance is already running.
        """

    @abstractmethod
    def release_single_instance_lock(self) -> None:
        """Release the single-instance lock."""

    @abstractmethod
    def get_config_dir(self) -> str:
        """Get the platform-appropriate configuration directory.

        Returns:
            Absolute path to the config directory
            (e.g. %APPDATA%/SwitchLang on Windows, ~/.config/switchlang on Linux).
        """

    @abstractmethod
    def is_startup_enabled(self, app_name: str) -> bool:
        """Check if the application is set to run on OS startup.

        Args:
            app_name: Application identifier.

        Returns:
            True if autostart is configured.
        """

    @abstractmethod
    def set_startup_enabled(self, enabled: bool, app_name: str) -> bool:
        """Enable or disable application launch on OS startup.

        Args:
            enabled: True to enable, False to disable.
            app_name: Application identifier.

        Returns:
            True on success, False on failure.
        """

    @abstractmethod
    def set_app_id(self, app_id: str) -> None:
        """Set the application identifier for the OS window manager.

        On Windows this sets AppUserModelID for taskbar grouping.
        On Linux this is a no-op (handled by .desktop files).

        Args:
            app_id: Application identifier string.
        """

    # =========================================================================
    # KEYCODE TRANSLATION
    # =========================================================================

    @abstractmethod
    def translate_keycode(self, native_keycode: int) -> int:
        """Translate a native OS keycode to a normalised virtual key code.

        The normalised keycodes match Windows VK_* values so the core
        pipeline can use a single set of constants.

        Args:
            native_keycode: The raw keycode from the OS event.

        Returns:
            Normalised virtual key code (Windows VK_* convention).
        """

    @abstractmethod
    def get_native_keycode(self, vk_code: int) -> int:
        """Translate a normalised VK code back to the native OS keycode.

        Args:
            vk_code: Normalised virtual key code (Windows VK_* convention).

        Returns:
            Native OS keycode for input injection.
        """
