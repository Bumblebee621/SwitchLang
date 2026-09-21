"""
SwitchLang — Real-time keyboard layout auto-switcher (EN ↔ HE).

Entry point: loads config, initializes all modules, starts hooks
and the PyQt6 system tray UI.
"""

import json
import logging
import logging.handlers
import os
import sys
import ctypes
import signal

# Global handle for the single-instance mutex
_mutex_handle = None

# Configure PyInstaller paths
BUNDLE_DIR = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))

# Keep user data in APPDATA (or ~/.config on non-Windows)
STORAGE_DIR = os.path.join(os.getenv('APPDATA') or os.path.expanduser('~/.config'), 'SwitchLang')
os.makedirs(STORAGE_DIR, exist_ok=True)

_LOG_FORMAT = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
_LOG_DATE_FORMAT = '%m-%d %H:%M:%S'
_log_file_handler = None   # Lazy-created when debug mode is enabled

logging.basicConfig(
    level=logging.WARNING,
    format=_LOG_FORMAT,
    datefmt=_LOG_DATE_FORMAT
)
logger = logging.getLogger('switchlang')


def set_debug_mode(enabled):
    """Toggle expressive logging (file + DEBUG level) on or off.

    When enabled is True, attaches a RotatingFileHandler to the root
    logger and drops level to DEBUG. When False, removes the file handler
    and restores WARNING level so disk output remains silent.
    """
    global _log_file_handler
    root = logging.getLogger()

    if enabled:
        if _log_file_handler is None:
            _log_file_handler = logging.handlers.RotatingFileHandler(
                os.path.join(STORAGE_DIR, 'switchlang.log'),
                maxBytes=200 * 1024, backupCount=1, encoding='utf-8'
            )
            _log_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
        if _log_file_handler not in root.handlers:
            root.addHandler(_log_file_handler)
        root.setLevel(logging.DEBUG)
    else:
        if _log_file_handler and _log_file_handler in root.handlers:
            root.removeHandler(_log_file_handler)
        root.setLevel(logging.WARNING)


from PyQt6.QtCore import QDir, QTimer
from PyQt6.QtWidgets import QApplication

from core.quadgram import load_models
from core.engine import EvaluationEngine
from core.sensitivity import SensitivityManager
from core.blacklist import BlacklistManager
from core.hooks import HookManager
from core import __version__
from ui.tray import SystemTrayApp
from ui.settings_window import SettingsWindow

CONFIG_PATH = os.path.join(STORAGE_DIR, 'config.json')
DATA_DIR = os.path.join(BUNDLE_DIR, 'data')
STYLE_PATH = os.path.join(BUNDLE_DIR, 'ui', 'style.qss')
COLLISIONS_PATH = os.path.join(DATA_DIR, 'collisions.json')


def load_config():
    """Load configuration from config.json."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def on_settings_changed(config_data, hook_manager):
    """Handle settings changes from the UI."""
    set_debug_mode(config_data.get('debug_mode', False))
    hook_manager.apply_config(config_data)


def main():
    """Application entry point."""
    # Prevent multiple instances using a named Windows Mutex (session-specific)
    mutex_name = "Local\\SwitchLang_Mutex_v1"
    ERROR_ALREADY_EXISTS = 183

    global _mutex_handle
    _mutex_handle = ctypes.windll.kernel32.CreateMutexW(None, False, mutex_name)
    last_error = ctypes.windll.kernel32.GetLastError()

    if last_error == ERROR_ALREADY_EXISTS:
        print("SwitchLang is already running.")
        sys.exit(0)

    # Set AppUserModelID so Windows taskbar groups windows by this ID instead of python.exe
    try:
        myappid = 'Bumblebee621.SwitchLang.v1'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass

    config = load_config()

    # Apply initial debug_mode from config
    debug = config.get('debug_mode', False)
    set_debug_mode(debug)

    try:
        models = load_models(DATA_DIR, load_so=True)
    except (FileNotFoundError, OSError, ValueError) as e:
        sys.exit(
            f"Error: Model files not found in {DATA_DIR} ({e}).\n"
            "Please run 'python scripts/build_quadgrams.py' to generate them."
        )

    engine = EvaluationEngine(
        models['en'], models['he'], COLLISIONS_PATH,
        storage_dir=STORAGE_DIR, enable_logging=debug,
        en_so_model=models.get('so'),
        model_mode=config.get('model_mode', 'standard')
    )

    sensitivity = SensitivityManager(
        baseline_delta=config.get('baseline_delta', 3.5),
        alpha=config.get('sensitivity_alpha', 0.3)
    )

    blacklist = BlacklistManager(CONFIG_PATH)

    hook_manager = HookManager(engine, sensitivity, blacklist, config)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    QDir.addSearchPath('ui', os.path.join(BUNDLE_DIR, 'ui'))
    if os.path.exists(STYLE_PATH):
        with open(STYLE_PATH, 'r', encoding='utf-8') as f:
            app.setStyleSheet(f.read())

    icon_path = os.path.join(DATA_DIR, 'icon.png')
    settings_window = SettingsWindow(CONFIG_PATH, blacklist, icon_path, version=__version__)

    tray = SystemTrayApp(settings_window, hook_manager, icon_path=icon_path)
    tray.show()

    settings_window.settings_changed.connect(
        lambda data: on_settings_changed(data, hook_manager)
    )
    settings_window.settings_changed.connect(
        tray.update_from_settings
    )

    # UI Notifications from Hooks
    hook_manager.set_on_suspend_callback(tray.notify_suspension)

    hook_manager.start()

    # Handle Ctrl+C gracefully
    def handle_sigint(sig, frame):
        logger.info("SIGINT received, shutting down...")
        QApplication.quit()

    signal.signal(signal.SIGINT, handle_sigint)

    # A QTimer allows Python to process signals (like SIGINT) while Qt event loop runs
    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    print('SwitchLang is running in the system tray.')
    print('Right-click the tray icon for options.')

    exit_code = app.exec()

    # Cleanup tray icon explicitly to avoid C++ runtime errors on shutdown
    tray.hide()
    tray.deleteLater()

    hook_manager.stop()

    if _mutex_handle:
        ctypes.windll.kernel32.CloseHandle(_mutex_handle)

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
