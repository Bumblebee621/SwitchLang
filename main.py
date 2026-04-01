"""
SwitchLang — Real-time keyboard layout auto-switcher (EN ↔ HE).

Entry point: loads config, initializes all modules, starts hooks
and the PyQt6 system tray UI.

Cross-platform: works on Windows and Linux (X11).
"""

import json
import logging
import logging.handlers
import os
import sys
import signal
from collections import deque

from core.platform import get_platform_backend

# Initialize platform backend early (needed for paths)
platform = get_platform_backend()

# Global handle for platform-specific single-instance lock
_platform_ref = platform

# Configure PyInstaller paths
if getattr(sys, 'frozen', False):
    BUNDLE_DIR = sys._MEIPASS
    APP_DIR = os.path.dirname(sys.executable)
else:
    BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
    APP_DIR = BUNDLE_DIR

# Keep user data in the platform-appropriate config directory
STORAGE_DIR = platform.get_config_dir()

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

# Custom formatter that trims the logger name to only its last segment
class _ShortNameFormatter(logging.Formatter):
    def format(self, record):
        record.shortname = record.name.rsplit('.', 1)[-1]
        return super().format(record)

_LOG_FORMAT = '%(asctime)s [%(shortname)s] %(levelname)s: %(message)s'
_LOG_DATE_FORMAT = '%m-%d %H:%M:%S'
_log_file_handler = None

_console_formatter = _ShortNameFormatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_console_formatter)

logging.basicConfig(
    level=logging.WARNING,
    handlers=[_console_handler]
)
logger = logging.getLogger('switchlang')


def set_debug_mode(enabled):
    """Toggle expressive logging (file + DEBUG level) on or off."""
    global _log_file_handler
    root = logging.getLogger()

    if enabled:
        if _log_file_handler is None:
            _log_file_handler = LineRotatingFileHandler(
                os.path.join(STORAGE_DIR, 'switchlang.log'),
                max_lines=1000, encoding='utf-8'
            )
            _log_file_handler.setFormatter(_ShortNameFormatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
        if _log_file_handler not in root.handlers:
            root.addHandler(_log_file_handler)
        root.setLevel(logging.DEBUG)
    else:
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
from core import startup as startup_module
from ui.tray import SystemTrayApp
from ui.settings_window import SettingsWindow

CONFIG_PATH = os.path.join(STORAGE_DIR, 'config.json')
DATA_DIR = os.path.join(BUNDLE_DIR, 'data')
STYLE_PATH = os.path.join(BUNDLE_DIR, 'ui', 'style.qss')
COLLISIONS_PATH = os.path.join(DATA_DIR, 'collisions.json')


def load_config():
    """Load configuration from config.json."""
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
    """Load the QSS stylesheet and fix relative resource paths."""
    if os.path.exists(STYLE_PATH):
        with open(STYLE_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
            
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
    """Handle settings changes from the UI."""
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
    global platform

    # Single-instance lock
    if not platform.set_single_instance_lock('SwitchLang'):
        print("SwitchLang is already running.")
        sys.exit(0)

    # Set app ID for window manager grouping
    platform.set_app_id('Bumblebee621.SwitchLang.v1')

    # Initialize startup module with platform backend
    startup_module.set_platform(platform)

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
    blacklist.set_platform(platform)

    hook_manager = HookManager(engine, sensitivity, blacklist, config, platform)

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

    try:
        hook_manager.start()
    except PermissionError as e:
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Permission Error")
        msg.setText("Cannot access the keyboard device.")
        msg.setInformativeText(str(e))
        msg.exec()
        sys.exit(1)

    # Handle Ctrl+C gracefully
    def handle_sigint(sig, frame):
        logger.info("SIGINT received, shutting down...")
        QApplication.quit()

    signal.signal(signal.SIGINT, handle_sigint)

    from PyQt6.QtCore import QTimer
    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    print('SwitchLang is running in the system tray.')
    print('Right-click the tray icon for options.')

    exit_code = app.exec()

    # Cleanup
    tray.hide()
    tray.deleteLater()

    hook_manager.stop()

    platform.release_single_instance_lock()

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
