"""
settings_window.py — Main settings dialog with General and Blacklist tabs.
"""

import json
import os

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QCheckBox, QSlider, QPushButton, QLineEdit,
    QListWidget, QGroupBox, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont


class SettingsWindow(QMainWindow):
    """Settings window accessible from the system tray."""

    settings_changed = pyqtSignal(dict)

    def __init__(self, config_path, blacklist_manager, parent=None):
        super().__init__(parent)
        self.config_path = config_path
        self.blacklist_manager = blacklist_manager

        self.setWindowTitle('SwitchLang — Settings')
        self.setFixedSize(520, 560)
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

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        save_btn = QPushButton('Save')
        save_btn.setObjectName('primary_button')
        save_btn.setFixedWidth(100)
        save_btn.clicked.connect(self._save_config)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

    def _build_general_tab(self):
        """Build the General settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)

        enable_group = QGroupBox('Engine Control')
        eg_layout = QVBoxLayout(enable_group)

        self.enable_check = QCheckBox('Enable auto-switching')
        self.enable_check.setFont(QFont('Segoe UI', 13))
        eg_layout.addWidget(self.enable_check)

        self.status_label = QLabel('Active')
        self.status_label.setObjectName('status_active')
        self.enable_check.toggled.connect(self._update_status_label)
        eg_layout.addWidget(self.status_label)

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
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(1, 50)
        self.sensitivity_slider.setTickInterval(5)
        self.sensitivity_slider.setTickPosition(
            QSlider.TickPosition.TicksBelow
        )
        self.sensitivity_slider.valueChanged.connect(
            self._update_delta_label
        )
        slider_row.addWidget(self.sensitivity_slider)

        self.delta_label = QLabel('Δ = 2.0')
        self.delta_label.setFixedWidth(60)
        self.delta_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.delta_label.setStyleSheet(
            'color: #89b4fa; font-weight: bold;'
        )
        slider_row.addWidget(self.delta_label)

        sg_layout.addLayout(slider_row)
        layout.addWidget(sense_group)

        layout.addStretch()
        return tab

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
        """Update the status indicator when the toggle changes."""
        if checked:
            self.status_label.setText('● Active')
            self.status_label.setObjectName('status_active')
        else:
            self.status_label.setText('● Inactive')
            self.status_label.setObjectName('status_inactive')
        self.status_label.setStyleSheet(
            self.status_label.styleSheet()
        )
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _update_delta_label(self, value):
        """Update the delta display when the slider moves."""
        delta = value / 10.0
        self.delta_label.setText(f'\u0394 = {delta:.1f}')

    def _load_config(self):
        """Load settings from config.json into the UI."""
        data = {}
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        self.enable_check.setChecked(data.get('enabled', True))
        delta = data.get('baseline_delta', 2.0)
        self.sensitivity_slider.setValue(int(delta * 10))

        self.blacklist_widget.clear()
        for exe in self.blacklist_manager.get_list():
            self.blacklist_widget.addItem(exe)

    def _save_config(self):
        """Save current UI state to config.json and emit signal."""
        data = {}
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        data['enabled'] = self.enable_check.isChecked()
        data['baseline_delta'] = self.sensitivity_slider.value() / 10.0

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        self.settings_changed.emit(data)
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
        self._fg_timer.stop()
        event.ignore()
        self.hide()
