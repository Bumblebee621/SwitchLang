"""
settings_window.py — Main settings dialog with General and Blacklist tabs.
"""

import json
import os

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QCheckBox, QSlider, QPushButton, QLineEdit,
    QListWidget, QGroupBox, QFrame, QSizePolicy, QMessageBox,
    QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

from core.startup import is_startup_enabled, set_startup_enabled


class NoWheelSlider(QSlider):
    """A QSlider that ignores wheel events."""
    def wheelEvent(self, event):
        event.ignore()


class SettingsWindow(QMainWindow):
    """Settings window accessible from the system tray."""

    settings_changed = pyqtSignal(dict)

    def __init__(self, config_path, blacklist_manager, parent=None):
        super().__init__(parent)
        self.config_path = config_path
        self.blacklist_manager = blacklist_manager

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

        # Debug Mode Group
        debug_group = QGroupBox('Advanced')
        dg_layout = QVBoxLayout(debug_group)

        self.debug_check = QCheckBox('Enable Debug Mode (Dangerous)')
        self.debug_check.setFont(QFont('Segoe UI', 11))
        self.debug_check.setStyleSheet('color: #f38ba8;')
        self.debug_check.toggled.connect(self._toggle_debug_mode)
        dg_layout.addWidget(self.debug_check)

        debug_desc = QLabel(
            'Records ALL keystrokes to the log file for troubleshooting. '
            'Passwords and private data may be recorded.'
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
