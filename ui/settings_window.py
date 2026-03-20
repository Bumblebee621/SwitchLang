"""
settings_window.py — Main settings dialog with General and Blacklist tabs.
"""

import json
import os

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QCheckBox, QSlider, QPushButton, QLineEdit,
    QListWidget, QGroupBox, QFrame, QSizePolicy, QMessageBox,
    QScrollArea, QSpinBox, QProgressBar, QDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt6.QtGui import QFont, QIcon, QKeyEvent

from core.startup import is_startup_enabled, set_startup_enabled
from core.updater import check_for_updates, download_and_install

# Map Windows VK codes to human-readable names
VK_NAME_MAP = {
    0x08: 'Backspace', 0x09: 'Tab', 0x0D: 'Enter', 0x1B: 'Esc',
    0x20: 'Space', 0x21: 'PgUp', 0x22: 'PgDn', 0x23: 'End', 0x24: 'Home',
    0x25: 'Left', 0x26: 'Up', 0x27: 'Right', 0x28: 'Down',
    0x2D: 'Insert', 0x2E: 'Delete',
    0x30: '0', 0x31: '1', 0x32: '2', 0x33: '3', 0x34: '4',
    0x35: '5', 0x36: '6', 0x37: '7', 0x38: '8', 0x39: '9',
    0x41: 'A', 0x42: 'B', 0x43: 'C', 0x44: 'D', 0x45: 'E',
    0x46: 'F', 0x47: 'G', 0x48: 'H', 0x49: 'I', 0x4A: 'J',
    0x4B: 'K', 0x4C: 'L', 0x4D: 'M', 0x4E: 'N', 0x4F: 'O',
    0x50: 'P', 0x51: 'Q', 0x52: 'R', 0x53: 'S', 0x54: 'T',
    0x55: 'U', 0x56: 'V', 0x57: 'W', 0x58: 'X', 0x59: 'Y', 0x5A: 'Z',
    0x60: 'Num0', 0x61: 'Num1', 0x62: 'Num2', 0x63: 'Num3', 0x64: 'Num4',
    0x65: 'Num5', 0x66: 'Num6', 0x67: 'Num7', 0x68: 'Num8', 0x69: 'Num9',
    0x6A: 'Num*', 0x6B: 'Num+', 0x6D: 'Num-', 0x6E: 'Num.', 0x6F: 'Num/',
    0x70: 'F1', 0x71: 'F2', 0x72: 'F3', 0x73: 'F4', 0x74: 'F5',
    0x75: 'F6', 0x76: 'F7', 0x77: 'F8', 0x78: 'F9', 0x79: 'F10',
    0x7A: 'F11', 0x7B: 'F12',
    0xFF: 'Fn', # Some laptop drivers map Fn here
    0xBA: ';', 0xBB: '=', 0xBC: ',', 0xBD: '-', 0xBE: '.', 0xBF: '/',
    0xC0: '`', 0xDB: '[', 0xDC: '\\', 0xDD: ']', 0xDE: "'",
}

VK_CONTROL = 0x11
VK_SHIFT = 0x10
VK_MENU = 0x12  # Alt


def vk_list_to_label(vk_list):
    """Convert a list of VK codes to a human-readable label like 'Ctrl+F12'."""
    if not vk_list:
        return 'Not set'
    parts = []
    modifiers_order = [(VK_CONTROL, 'Ctrl'), (VK_SHIFT, 'Shift'), (VK_MENU, 'Alt')]
    remaining = set(vk_list)
    for vk, name in modifiers_order:
        if vk in remaining:
            parts.append(name)
            remaining.discard(vk)
    for vk in sorted(remaining):
        parts.append(VK_NAME_MAP.get(vk, f'0x{vk:02X}'))
    return ' + '.join(parts)


class NoWheelSlider(QSlider):
    """A QSlider that ignores wheel events."""
    def wheelEvent(self, event):
        event.ignore()

class NoWheelSpinBox(QSpinBox):
    """A QSpinBox that ignores wheel events."""
    def wheelEvent(self, event):
        event.ignore()


class UpdateWorker(QThread):
    """Worker thread to check for updates without freezing UI."""
    finished = pyqtSignal(str, str) # version, url

    def run(self):
        v, url = check_for_updates()
        if v and url:
            self.finished.emit(v, url)
        else:
            self.finished.emit("", "")

class DownloadWorker(QThread):
    """Worker thread to download and install updates."""
    progress = pyqtSignal(int, int)
    error = pyqtSignal(str)

    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):
        try:
            download_and_install(self.url, progress_callback=self.progress.emit)
        except Exception as e:
            self.error.emit(str(e))

class ProgressDialog(QDialog):
    """Simple dialog showing download progress."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Downloading Update")
        self.setFixedSize(300, 100)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        
        layout = QVBoxLayout(self)
        self.label = QLabel("Downloading SwitchLang...")
        layout.addWidget(self.label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

    def set_progress(self, current, total):
        if total > 0:
            val = int((current / total) * 100)
            self.progress_bar.setValue(val)


class SettingsWindow(QMainWindow):
    """Settings window accessible from the system tray."""

    settings_changed = pyqtSignal(dict)

    def __init__(self, config_path, blacklist_manager, icon_path, version="1.0.0", parent=None):
        super().__init__(parent)
        self.config_path = config_path
        self.blacklist_manager = blacklist_manager
        self.version = version

        if icon_path and os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Guard: timer is created inside _build_blacklist_tab; initialise
        # to None here so closeEvent is safe even if construction fails.
        self._fg_timer = None

        self.setWindowTitle('SwitchLang — Settings')
        self.setMinimumSize(520, 560)
        self.setWindowFlags(
            Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )

        self._build_ui()
        self._load_config()

    def _build_ui(self):
        """Build the full UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        title = QLabel('SwitchLang')
        title.setObjectName('title_label')
        layout.addWidget(title)

        subtitle = QLabel(
            'Real-time keyboard layout auto-switcher for '
            'English \u2194 Hebrew'
        )
        subtitle.setObjectName('subtitle_label')
        layout.addWidget(subtitle)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet('background-color: #45475a; max-height: 1px;')
        layout.addWidget(sep)

        tabs = QTabWidget()
        tabs.addTab(self._build_general_tab(), 'General')
        tabs.addTab(self._build_blacklist_tab(), 'Blacklist')
        tabs.addTab(self._build_about_tab(), 'About')
        layout.addWidget(tabs)


    def _build_general_tab(self):
        """Build the General settings tab with scrolling support."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        # Ensure the scroll area doesn't have a background that clashes
        scroll.setStyleSheet('background: transparent;')

        container = QWidget()
        container.setObjectName('general_tab_container')
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        enable_group = QGroupBox('Engine Control')
        eg_layout = QVBoxLayout(enable_group)

        self.enable_check = QCheckBox('Enable auto-switching')
        self.enable_check.setFont(QFont('Segoe UI', 13))
        eg_layout.addWidget(self.enable_check)

        self.status_label = QLabel('Active')
        self.status_label.setObjectName('status_active')
        self.enable_check.toggled.connect(self._update_status_label)
        self.enable_check.toggled.connect(self._apply_settings)
        eg_layout.addWidget(self.status_label)

        self.startup_check = QCheckBox('Launch SwitchLang on startup')
        self.startup_check.setFont(QFont('Segoe UI', 11))
        self.startup_check.setChecked(is_startup_enabled())
        self.startup_check.toggled.connect(self._toggle_startup)
        eg_layout.addWidget(self.startup_check)

        layout.addWidget(enable_group)

        sense_group = QGroupBox('Sensitivity')
        sg_layout = QVBoxLayout(sense_group)
        
        desc = QLabel(
            'Lower values = more aggressive switching.\n'
            'Higher values = more conservative (fewer false positives).'
        )
        desc.setWordWrap(True)
        desc.setStyleSheet('color: #6c7086; font-size: 11px;')
        sg_layout.addWidget(desc)

        slider_row = QHBoxLayout()
        self.sensitivity_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(1, 80)
        self.sensitivity_slider.setTickInterval(5)
        self.sensitivity_slider.setTickPosition(
            QSlider.TickPosition.TicksBelow
        )
        self.sensitivity_slider.valueChanged.connect(
            self._update_delta_label
        )
        self.sensitivity_slider.valueChanged.connect(
            self._apply_settings
        )
        slider_row.addWidget(self.sensitivity_slider)

        # I1: initialise label from slider's actual default, not a hardcoded string.
        _default_delta = self.sensitivity_slider.value() / 10.0
        self.delta_label = QLabel(f'\u0394 = {_default_delta:.1f}')
        self.delta_label.setFixedWidth(60)
        self.delta_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.delta_label.setStyleSheet(
            'color: #89b4fa; font-weight: bold;'
        )
        slider_row.addWidget(self.delta_label)

        sg_layout.addLayout(slider_row)
        layout.addWidget(sense_group)

        # Suspension Group
        suspend_group = QGroupBox('Suspend Hotkey')
        susp_layout = QVBoxLayout(suspend_group)

        susp_desc = QLabel(
            'Press a key combination to temporarily pause auto-switching.\n'
            'Press the same hotkey again to resume early.'
        )
        susp_desc.setWordWrap(True)
        susp_desc.setStyleSheet('color: #6c7086; font-size: 11px;')
        susp_layout.addWidget(susp_desc)

        key_row = QHBoxLayout()
        key_row.addWidget(QLabel('Keybind:'))

        self._suspend_vks = []  # Current recorded VK list
        self._recording_keybind = False

        self.keybind_btn = QPushButton('Not set')
        self.keybind_btn.setFixedHeight(30)
        self.keybind_btn.clicked.connect(self._start_keybind_recording)
        key_row.addWidget(self.keybind_btn)

        clear_btn = QPushButton('Clear keybind')
        clear_btn.setFixedHeight(30)
        clear_btn.setToolTip('Clear keybind')
        clear_btn.clicked.connect(self._clear_keybind)
        key_row.addWidget(clear_btn)

        susp_layout.addLayout(key_row)

        dur_row = QHBoxLayout()
        dur_row.addWidget(QLabel('Duration (seconds):'))
        self.suspend_duration_spin = NoWheelSpinBox()
        self.suspend_duration_spin.setRange(5, 300)
        self.suspend_duration_spin.setValue(60)
        self.suspend_duration_spin.setSuffix(' s')
        self.suspend_duration_spin.valueChanged.connect(self._apply_settings)
        dur_row.addWidget(self.suspend_duration_spin)
        susp_layout.addLayout(dur_row)

        layout.addWidget(suspend_group)

        # Language Model Group
        model_group = QGroupBox('Language Model')
        mg_layout = QVBoxLayout(model_group)

        model_desc = QLabel(
            'Standard: General conversational context.\n'
            'Smart: Technical mode in IDEs & Editors, Standard elsewhere.\n'
            'Always Technical: Enhanced for programming terms everywhere.'
        )
        model_desc.setWordWrap(True)
        model_desc.setStyleSheet('color: #6c7086; font-size: 11px;')
        mg_layout.addWidget(model_desc)

        self.model_std_radio = QCheckBox('Standard (Conversational)')
        self.model_smart_radio = QCheckBox('Smart (IDEs & Editors)')
        self.model_tech_radio = QCheckBox('Always Technical')
        
        # Make them behave like radio buttons
        self.model_std_radio.toggled.connect(self._on_model_std_toggled)
        self.model_smart_radio.toggled.connect(self._on_model_smart_toggled)
        self.model_tech_radio.toggled.connect(self._on_model_tech_toggled)
        
        mg_layout.addWidget(self.model_std_radio)
        mg_layout.addWidget(self.model_smart_radio)
        mg_layout.addWidget(self.model_tech_radio)

        layout.addWidget(model_group)

        # Debug Mode Group
        debug_group = QGroupBox('Advanced')
        dg_layout = QVBoxLayout(debug_group)

        self.debug_check = QCheckBox('Enable Debug Mode (Dangerous)')
        self.debug_check.setFont(QFont('Segoe UI', 11))
        self.debug_check.setStyleSheet('color: #f38ba8;')
        self.debug_check.toggled.connect(self._toggle_debug_mode)
        dg_layout.addWidget(self.debug_check)

        STORAGE_DIR = os.path.join(os.getenv('APPDATA'), 'SwitchLang')
        debug_desc = QLabel(
            'Records ALL keystrokes to the log file for troubleshooting.\n'
            'Passwords and private data may be recorded.\n'
            f'.SCV and .log files can be found in {STORAGE_DIR}'
        )
        debug_desc.setWordWrap(True)
        debug_desc.setStyleSheet('color: #6c7086; font-size: 10px;')
        dg_layout.addWidget(debug_desc)

        layout.addWidget(debug_group)

        layout.addStretch()
        
        scroll.setWidget(container)
        return scroll

    def _build_blacklist_tab(self):
        """Build the Blacklist manager tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)

        desc = QLabel(
            'Applications in this list will not trigger '
            'automatic layout switching.'
        )
        desc.setWordWrap(True)
        desc.setStyleSheet('color: #6c7086; font-size: 11px;')
        layout.addWidget(desc)

        self.blacklist_widget = QListWidget()
        layout.addWidget(self.blacklist_widget)

        add_row = QHBoxLayout()
        self.exe_input = QLineEdit()
        self.exe_input.setPlaceholderText('e.g. code.exe')
        self.exe_input.returnPressed.connect(self._add_exe)
        add_row.addWidget(self.exe_input)

        add_btn = QPushButton('Add')
        add_btn.setObjectName('primary_button')
        add_btn.setFixedWidth(60)
        add_btn.clicked.connect(self._add_exe)
        add_row.addWidget(add_btn)
        layout.addLayout(add_row)

        action_row = QHBoxLayout()

        remove_btn = QPushButton('Remove Selected')
        remove_btn.setObjectName('danger_button')
        remove_btn.clicked.connect(self._remove_exe)
        action_row.addWidget(remove_btn)

        action_row.addStretch()

        self.capture_btn = QPushButton('⊕ Blacklist: ...')
        self.capture_btn.setToolTip(
            'Add the currently focused application to the blacklist'
        )
        self.capture_btn.clicked.connect(self._capture_foreground)
        action_row.addWidget(self.capture_btn)

        self._fg_timer = QTimer(self)
        self._fg_timer.setInterval(1000)
        self._fg_timer.timeout.connect(self._update_capture_btn)

        layout.addLayout(action_row)
        return tab

    def _build_about_tab(self):
        """Build the About/Update tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 40, 30, 40)

        # Version Info
        version_label = QLabel(f'Version {self.version}')
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setStyleSheet('font-size: 16px; font-weight: bold; color: #89b4fa;')
        layout.addWidget(version_label)

        author_label = QLabel('Created by Ariel')
        author_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author_label.setStyleSheet('color: #6c7086;')
        layout.addWidget(author_label)

        layout.addSpacing(20)

        # Update Section
        self.update_btn = QPushButton('Check for Updates')
        self.update_btn.setObjectName('primary_button')
        self.update_btn.setFixedHeight(40)
        self.update_btn.clicked.connect(self._check_updates)
        layout.addWidget(self.update_btn)

        self.update_status = QLabel('')
        self.update_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.update_status.setStyleSheet('color: #a6adc8; font-size: 12px;')
        layout.addWidget(self.update_status)

        layout.addStretch()

        # GitHub Link
        github_label = QLabel('<a href="https://github.com/Bumblebee621/SwitchLang" style="color: #89b4fa; text-decoration: none;">GitHub Repository</a>')
        github_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        github_label.setOpenExternalLinks(True)
        layout.addWidget(github_label)

        return tab

    def _check_updates(self):
        """Start the background update check."""
        self.update_btn.setEnabled(False)
        self.update_btn.setText("Checking...")
        self.update_status.setText("")
        
        self._update_worker = UpdateWorker()
        self._update_worker.finished.connect(self._on_update_check_finished)
        self._update_worker.start()

    def _on_update_check_finished(self, version, url):
        """Handle the result of the update check."""
        self.update_btn.setEnabled(True)
        self.update_btn.setText("Check for Updates")
        
        if not version:
            self.update_status.setText("You are using the latest version.")
            return

        # Update available
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Update Available")
        msg.setText(f"A new version is available: v{version}")
        msg.setInformativeText("Would you like to download and install it now?")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.setDefaultButton(QMessageBox.StandardButton.Yes)
        
        if msg.exec() == QMessageBox.StandardButton.Yes:
            self._start_download(url)

    def _start_download(self, url):
        """Start downloading the update."""
        self._progress_dialog = ProgressDialog(self)
        self._progress_dialog.show()
        
        self._download_worker = DownloadWorker(url)
        self._download_worker.progress.connect(self._progress_dialog.set_progress)
        self._download_worker.error.connect(self._on_download_error)
        self._download_worker.start()

    def _on_download_error(self, err_msg):
        """Handle download errors."""
        self._progress_dialog.close()
        QMessageBox.critical(self, "Update Error", f"Failed to download update: {err_msg}")

    def _update_status_label(self, checked):
        """Update the status indicator when the toggle changes.

        B3: Use direct setStyleSheet instead of objectName + unpolish/polish,
        which is unreliable in PyQt6 when the object name changes at runtime.
        """
        if checked:
            self.status_label.setText('● Active')
            self.status_label.setStyleSheet('''
                color: #a6e3a1;
                background-color: rgba(166, 227, 161, 0.15);
                border: 1px solid rgba(166, 227, 161, 0.3);
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
            ''')
        else:
            self.status_label.setText('● Inactive')
            self.status_label.setStyleSheet('''
                color: #f38ba8;
                background-color: rgba(243, 139, 168, 0.15);
                border: 1px solid rgba(243, 139, 168, 0.3);
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
            ''')

    def _update_delta_label(self, value):
        """Update the delta display when the slider moves."""
        delta = value / 10.0
        self.delta_label.setText(f'\u0394 = {delta:.1f}')

    def _load_config(self):
        """Load settings from config.json into the UI."""
        data = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}

        self.enable_check.setChecked(data.get('enabled', True))
        self._update_status_label(self.enable_check.isChecked())
        self.debug_check.setChecked(data.get('debug_mode', False))
        delta = data.get('baseline_delta', 2.0)
        self.sensitivity_slider.setValue(int(delta * 10))

        # Model Mode
        mode = data.get('model_mode', 'standard')
        self.model_std_radio.blockSignals(True)
        self.model_smart_radio.blockSignals(True)
        self.model_tech_radio.blockSignals(True)
        
        self.model_std_radio.setChecked(mode == 'standard')
        self.model_smart_radio.setChecked(mode == 'smart')
        self.model_tech_radio.setChecked(mode == 'technical')
        
        self.model_std_radio.blockSignals(False)
        self.model_smart_radio.blockSignals(False)
        self.model_tech_radio.blockSignals(False)

        # Suspension
        self._suspend_vks = data.get('suspend_keybind_vks', [])
        self.keybind_btn.setText(vk_list_to_label(self._suspend_vks))
        self.suspend_duration_spin.setValue(data.get('suspend_duration_sec', 60))

        self.blacklist_widget.clear()
        for exe in self.blacklist_manager.get_list():
            self.blacklist_widget.addItem(exe)

    def _toggle_startup(self, checked):
        """Toggle the application's launch on startup setting."""
        set_startup_enabled(checked)

    def _apply_settings(self):
        """Save current UI state to config.json and emit signal automatically."""
        data = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}

        data['enabled'] = self.enable_check.isChecked()
        data['debug_mode'] = self.debug_check.isChecked()
        data['baseline_delta'] = self.sensitivity_slider.value() / 10.0
        data['suspend_keybind_vks'] = self._suspend_vks
        data['suspend_duration_sec'] = self.suspend_duration_spin.value()
        
        if self.model_tech_radio.isChecked():
            data['model_mode'] = 'technical'
        elif self.model_smart_radio.isChecked():
            data['model_mode'] = 'smart'
        else:
            data['model_mode'] = 'standard'

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        self.settings_changed.emit(data)

    def _toggle_debug_mode(self, checked):
        """Handle debug mode toggle with a warning prompt."""
        # If we are trying to enable it, show warning
        if checked:
            # We temporarily disconnect to avoid recursion if we need to uncheck it
            self.debug_check.blockSignals(True)
            
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Security Warning")
            msg.setText("Enabling Debug Mode is UNSAFE.")
            msg.setInformativeText(
                "In this mode, ALL keystrokes (including passwords) are recorded to the log file.\n\n"
                "Are you sure you want to proceed?"
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msg.setDefaultButton(QMessageBox.StandardButton.No)
            
            ret = msg.exec()
            
            if ret == QMessageBox.StandardButton.No:
                self.debug_check.setChecked(False)
                self.debug_check.blockSignals(False)
                return
                
            self.debug_check.blockSignals(False)
            
        self._apply_settings()

    def _on_model_std_toggled(self, checked):
        """Simulate radio button behavior for model selection."""
        if checked:
            self.model_smart_radio.blockSignals(True)
            self.model_smart_radio.setChecked(False)
            self.model_smart_radio.blockSignals(False)
            self.model_tech_radio.blockSignals(True)
            self.model_tech_radio.setChecked(False)
            self.model_tech_radio.blockSignals(False)
            self._apply_settings()
        elif not self.model_tech_radio.isChecked() and not self.model_smart_radio.isChecked():
            # Don't allow unchecking all
            self.model_std_radio.blockSignals(True)
            self.model_std_radio.setChecked(True)
            self.model_std_radio.blockSignals(False)

    def _on_model_smart_toggled(self, checked):
        """Simulate radio button behavior for model selection."""
        if checked:
            self.model_std_radio.blockSignals(True)
            self.model_std_radio.setChecked(False)
            self.model_std_radio.blockSignals(False)
            self.model_tech_radio.blockSignals(True)
            self.model_tech_radio.setChecked(False)
            self.model_tech_radio.blockSignals(False)
            self._apply_settings()
        elif not self.model_std_radio.isChecked() and not self.model_tech_radio.isChecked():
            self.model_smart_radio.blockSignals(True)
            self.model_smart_radio.setChecked(True)
            self.model_smart_radio.blockSignals(False)

    def _on_model_tech_toggled(self, checked):
        """Simulate radio button behavior for model selection."""
        if checked:
            self.model_std_radio.blockSignals(True)
            self.model_std_radio.setChecked(False)
            self.model_std_radio.blockSignals(False)
            self.model_smart_radio.blockSignals(True)
            self.model_smart_radio.setChecked(False)
            self.model_smart_radio.blockSignals(False)
            self._apply_settings()
        elif not self.model_std_radio.isChecked() and not self.model_smart_radio.isChecked():
            self.model_tech_radio.blockSignals(True)
            self.model_tech_radio.setChecked(True)
            self.model_tech_radio.blockSignals(False)

    def _start_keybind_recording(self):
        """Enter the recording state to capture a new hotkey."""
        self._recording_keybind = True
        self.keybind_btn.setText('Recording...')
        self.keybind_btn.setStyleSheet('background-color: #f9e2af; color: #11111b; font-weight: bold;')
        self.setFocus() # Ensure window captures keys

    def _clear_keybind(self):
        """Reset the suspension hotkey."""
        self._suspend_vks = []
        self.keybind_btn.setText('Not set')
        self.keybind_btn.setStyleSheet('')
        self._apply_settings()

    def keyPressEvent(self, event: QKeyEvent):
        """Intercept keys while recording to capture the suspension hotkey."""
        if not self._recording_keybind:
            super().keyPressEvent(event)
            return

        key = event.key()
        
        # Don't capture modifiers on their own as the final trigger
        if key in (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Meta):
            return

        # Map Qt key + modifiers to VK codes
        vks = []
        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier:
            vks.append(VK_CONTROL)
        if mods & Qt.KeyboardModifier.ShiftModifier:
            vks.append(VK_SHIFT)
        if mods & Qt.KeyboardModifier.AltModifier:
            vks.append(VK_MENU)
        
        # Map the primary key using native virtual key (layout-independent on Windows)
        vk = event.nativeVirtualKey()
        if vk:
            vks.append(vk)
        else:
            # Fallback for keys where nativeVirtualKey might be 0
            vks.append(key & 0xFF)

        self._suspend_vks = vks
        self._recording_keybind = False
        self.keybind_btn.setText(vk_list_to_label(vks))
        self.keybind_btn.setStyleSheet('')
        self._apply_settings()
        self.clearFocus()

    def _save_config(self):
        """Deprecated: Use _apply_settings instead."""
        self._apply_settings()
        self.close()

    def _add_exe(self):
        """Add a typed exe name to the blacklist."""
        exe = self.exe_input.text().strip()
        if exe:
            if not exe.lower().endswith('.exe'):
                exe += '.exe'
            self.blacklist_manager.add(exe)
            self._refresh_blacklist()
            self.exe_input.clear()

    def _remove_exe(self):
        """Remove the selected exe from the blacklist."""
        item = self.blacklist_widget.currentItem()
        if item:
            self.blacklist_manager.remove(item.text())
            self._refresh_blacklist()

    def _update_capture_btn(self):
        """Update the capture button with the current foreground app."""
        exe = self.blacklist_manager.get_foreground_exe()
        skip = {'python.exe', 'pythonw.exe', 'switchlang.exe'}
        if exe and exe not in skip:
            self._last_external_exe = exe
        if hasattr(self, '_last_external_exe') and self._last_external_exe:
            self.capture_btn.setText(
                f'⊕ Blacklist: {self._last_external_exe}'
            )
        else:
            self.capture_btn.setText('⊕ Blacklist Current App')

    def _capture_foreground(self):
        """Blacklist the last non-SwitchLang foreground app."""
        exe = getattr(self, '_last_external_exe', None)
        if exe:
            self.blacklist_manager.add(exe)
            self._refresh_blacklist()

    def _refresh_blacklist(self):
        """Reload the blacklist widget from the manager."""
        self.blacklist_widget.clear()
        for exe in self.blacklist_manager.get_list():
            self.blacklist_widget.addItem(exe)

    def showEvent(self, event):
        """Start the foreground app polling timer."""
        super().showEvent(event)
        self._update_capture_btn()
        self._fg_timer.start()

    def closeEvent(self, event):
        """Hide window and stop timer instead of closing."""
        if self._fg_timer is not None:
            self._fg_timer.stop()
        event.ignore()
        self.hide()
