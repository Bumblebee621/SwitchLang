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
from ctypes import wintypes
from collections import deque
 
# Global handle for the single-instance mutex
_mutex_handle = None

# Configure PyInstaller paths
if getattr(sys, 'frozen', False):
    BUNDLE_DIR = sys._MEIPASS
    APP_DIR = os.path.dirname(sys.executable)
else:
    BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
    APP_DIR = BUNDLE_DIR

# Keep user data in APPDATA to avoid cluttering the repository
STORAGE_DIR = os.path.join(os.getenv('APPDATA'), 'SwitchLang')

# Ensure storage directory exists
os.makedirs(STORAGE_DIR, exist_ok=True)

class LineRotatingFileHandler(logging.Handler):
    """A log handler that limits the file to a maximum number of lines."""
    def __init__(self, filename, max_lines=1000, encoding='utf-8'):
        super().__init__()
        self.filename = filename
        self.encoding = encoding
        self.lines = deque(maxlen=max_lines)
        if os.path.exists(self.filename):
            with open(self.filename, 'r', encoding=self.encoding) as f:
                self.lines.extend(f.readlines())
                
    def emit(self, record):
        try:
            msg = self.format(record)
            self.lines.append(msg + '\n')
            with open(self.filename, 'w', encoding=self.encoding) as f:
                f.writelines(self.lines)
        except Exception:
            self.handleError(record)

# Configure logging — console-only by default (WARNING level).
# Debug mode (toggled by the user) adds a file handler and switches to DEBUG.
_LOG_FORMAT = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
_log_file_handler = None   # Lazy-created when debug mode is enabled

logging.basicConfig(
    level=logging.WARNING,
    format=_LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('switchlang')


def set_debug_mode(enabled):
    """Toggle expressive logging (file + DEBUG level) on or off.

    When *enabled* is True, attaches a LineRotatingFileHandler to the root
    logger and drops every ``switchlang.*`` logger to DEBUG.  When False,
    removes the file handler and restores WARNING level so the app stays
    completely silent on disk.
    """
    global _log_file_handler
    root = logging.getLogger()

    if enabled:
        # Attach file handler (once)
        if _log_file_handler is None:
            _log_file_handler = LineRotatingFileHandler(
                os.path.join(STORAGE_DIR, 'switchlang.log'),
                max_lines=1000, encoding='utf-8'
            )
            _log_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        if _log_file_handler not in root.handlers:
            root.addHandler(_log_file_handler)
        root.setLevel(logging.DEBUG)
    else:
        # Remove file handler and silence disk output
        if _log_file_handler and _log_file_handler in root.handlers:
            root.removeHandler(_log_file_handler)
        root.setLevel(logging.WARNING)

from PyQt6.QtWidgets import QApplication

from core.quadgram import load_models
from core.engine import EvaluationEngine
from core.sensitivity import SensitivityManager
from core.blacklist import BlacklistManager, DEFAULT_BLACKLIST
from core.hooks import HookManager
from core.version import __version__
from ui.tray import SystemTrayApp
from ui.settings_window import SettingsWindow

CONFIG_PATH = os.path.join(STORAGE_DIR, 'config.json')
DATA_DIR = os.path.join(BUNDLE_DIR, 'data')
STYLE_PATH = os.path.join(BUNDLE_DIR, 'ui', 'style.qss')
COLLISIONS_PATH = os.path.join(DATA_DIR, 'collisions.json')


def load_config():
    """Load configuration from config.json.

    Returns:
        Dict of configuration values.
    """
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'enabled': True,
        'baseline_delta': 2.0,
        'sensitivity_alpha': 0.3,
        'idle_timeout_seconds': 5.0,
        'debug_mode': False,
        'blacklist': sorted(list(DEFAULT_BLACKLIST))
    }


def load_stylesheet():
    """Load the QSS stylesheet and fix relative resource paths.

    Returns:
        QSS string or empty string if file not found.
    """
    if os.path.exists(STYLE_PATH):
        with open(STYLE_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Fix relative resource paths for PyInstaller bundling.
            # Convert url("ui/...") and url("data/...") to use absolute paths 
            # based on BUNDLE_DIR so they resolve correctly even when frozen.
            # We use forward slashes because QSS expects them even on Windows.
            safe_bundle_dir = BUNDLE_DIR.replace('\\', '/')
            content = content.replace('url("ui/', f'url("{safe_bundle_dir}/ui/')
            content = content.replace('url("data/', f'url("{safe_bundle_dir}/data/')
            
            return content
    return ''


def check_data_files():
    """Check that quadgram data files exist. If not, generate them."""
    en_path = os.path.join(DATA_DIR, 'en_quadgrams.json')
    he_path = os.path.join(DATA_DIR, 'he_quadgrams.json')

    if not os.path.exists(en_path) or not os.path.exists(he_path):
        print('Quadgram data files not found. Generating...')
        scripts_dir = os.path.join(APP_DIR, 'scripts')
        sys.path.insert(0, scripts_dir)
        from build_quadgrams import main as build_main
        build_main()
        sys.path.pop(0)
        print()


def on_settings_changed(config_data, hook_manager, sensitivity, engine):
    """Handle settings changes from the UI.

    Args:
        config_data: New configuration dict.
        hook_manager: HookManager instance.
        sensitivity: SensitivityManager instance.
        engine: EvaluationEngine instance.
    """
    debug = config_data.get('debug_mode', False)
    set_debug_mode(debug)
    hook_manager.set_enabled(config_data.get('enabled', True))
    hook_manager.set_debug_mode(debug)
    sensitivity.update_config(
        baseline_delta=config_data.get('baseline_delta', 2.0),
        alpha=config_data.get('sensitivity_alpha', 0.3)
    )
    hook_manager.idle_timeout = config_data.get(
        'idle_timeout_seconds', 5.0
    )
    # Suspension Config
    hook_manager.set_suspend_config(
        config_data.get('suspend_keybind_vks', []),
        config_data.get('suspend_duration_sec', 60)
    )
    # Model Mode
    mode = config_data.get('model_mode', 'standard')
    engine.set_model_mode(mode)
    hook_manager.set_model_mode(mode)


def main():
    """Application entry point."""
    # Prevent multiple instances using a named Windows Mutex
    # We prefix with 'Local\' to ensure it's session-specific.
    mutex_name = "Local\\SwitchLang_Mutex_v1"
    ERROR_ALREADY_EXISTS = 183
    
    # We must keep a reference to this handle for the duration of the app
    global _mutex_handle
    _mutex_handle = ctypes.windll.kernel32.CreateMutexW(None, False, mutex_name)
    last_error = ctypes.windll.kernel32.GetLastError()
    
    if last_error == ERROR_ALREADY_EXISTS:
        print("SwitchLang is already running.")
        sys.exit(0)

    # Set AppUserModelID so Windows taskbar groups windows by this ID instead of python.exe
    try:
        myappid = u'Bumblebee621.SwitchLang.v1' 
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass

    check_data_files()

    config = load_config()

    # Apply initial debug_mode from config
    debug = config.get('debug_mode', False)
    set_debug_mode(debug)

    models = load_models(DATA_DIR, load_so=True)

    engine = EvaluationEngine(
        models['en'], models['he'], COLLISIONS_PATH,
        storage_dir=STORAGE_DIR, enable_logging=debug,
        en_so_model=models.get('so'),
        model_mode=config.get('model_mode', 'standard')
    )

    sensitivity = SensitivityManager(
        baseline_delta=config.get('baseline_delta', 2.0),
        alpha=config.get('sensitivity_alpha', 0.3)
    )

    blacklist = BlacklistManager(CONFIG_PATH)

    hook_manager = HookManager(engine, sensitivity, blacklist, config)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    stylesheet = load_stylesheet()
    if stylesheet:
        app.setStyleSheet(stylesheet)

    icon_path = os.path.join(DATA_DIR, 'icon.png')
    settings_window = SettingsWindow(CONFIG_PATH, blacklist, icon_path, version=__version__)

    tray = SystemTrayApp(settings_window, hook_manager, icon_path=icon_path)
    tray.show()

    settings_window.settings_changed.connect(
        lambda data: on_settings_changed(data, hook_manager, sensitivity, engine)
    )
    settings_window.settings_changed.connect(
        tray.update_from_settings
    )

    # UI Notifications from Hooks
    hook_manager.set_on_suspend_callback(tray.notify_suspension)

    hook_manager.start()

    print('SwitchLang is running in the system tray.')
    print('Right-click the tray icon for options.')

    exit_code = app.exec()

    hook_manager.stop()

    if _mutex_handle:
        ctypes.windll.kernel32.CloseHandle(_mutex_handle)

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
