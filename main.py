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
from collections import deque

# Configure PyInstaller paths
if getattr(sys, 'frozen', False):
    BUNDLE_DIR = sys._MEIPASS
    APP_DIR = os.path.dirname(sys.executable)
else:
    BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
    APP_DIR = BUNDLE_DIR

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

# Configure logging — INFO to file, INFO to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        LineRotatingFileHandler(
            os.path.join(APP_DIR, 'switchlang.log'),
            max_lines=1000, encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger('switchlang.hooks').setLevel(logging.INFO)
logger = logging.getLogger('switchlang')

from PyQt6.QtWidgets import QApplication

from core.trigram import load_models
from core.engine import EvaluationEngine
from core.sensitivity import SensitivityManager
from core.blacklist import BlacklistManager, DEFAULT_BLACKLIST
from core.hooks import HookManager
from ui.tray import SystemTrayApp
from ui.settings_window import SettingsWindow

CONFIG_PATH = os.path.join(APP_DIR, 'config.json')
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
        'blacklist': sorted(list(DEFAULT_BLACKLIST))
    }


def load_stylesheet():
    """Load the QSS stylesheet.

    Returns:
        QSS string or empty string if file not found.
    """
    if os.path.exists(STYLE_PATH):
        with open(STYLE_PATH, 'r', encoding='utf-8') as f:
            return f.read()
    return ''


def check_data_files():
    """Check that trigram data files exist. If not, generate them."""
    en_path = os.path.join(DATA_DIR, 'en_trigrams.json')
    he_path = os.path.join(DATA_DIR, 'he_trigrams.json')

    if not os.path.exists(en_path) or not os.path.exists(he_path):
        print('Trigram data files not found. Generating...')
        scripts_dir = os.path.join(APP_DIR, 'scripts')
        sys.path.insert(0, scripts_dir)
        from build_trigrams import main as build_main
        build_main()
        sys.path.pop(0)
        print()


def on_settings_changed(config_data, hook_manager, sensitivity):
    """Handle settings changes from the UI.

    Args:
        config_data: New configuration dict.
        hook_manager: HookManager instance.
        sensitivity: SensitivityManager instance.
    """
    hook_manager.set_enabled(config_data.get('enabled', True))
    sensitivity.update_config(
        baseline_delta=config_data.get('baseline_delta', 2.0),
        alpha=config_data.get('sensitivity_alpha', 0.3)
    )
    hook_manager.idle_timeout = config_data.get(
        'idle_timeout_seconds', 5.0
    )


def main():
    """Application entry point."""
    check_data_files()

    config = load_config()

    en_model, he_model = load_models(DATA_DIR)

    engine = EvaluationEngine(en_model, he_model, COLLISIONS_PATH)

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

    settings_window = SettingsWindow(CONFIG_PATH, blacklist)

    tray = SystemTrayApp(settings_window, hook_manager)
    tray.show()

    settings_window.settings_changed.connect(
        lambda data: on_settings_changed(data, hook_manager, sensitivity)
    )
    settings_window.settings_changed.connect(
        tray.update_from_settings
    )

    hook_manager.start()

    print('SwitchLang is running in the system tray.')
    print('Right-click the tray icon for options.')

    exit_code = app.exec()

    hook_manager.stop()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
