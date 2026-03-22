"""
tray.py — System tray icon and context menu.
"""


import webbrowser
from PyQt6.QtWidgets import QSystemTrayIcon, QMenu, QApplication
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QAction
from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
import os


def _create_tray_icon_pixmap(suspended=False):
    """Create a programmatic 'SL' icon for the system tray."""
    size = 64
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(QColor(0, 0, 0, 0))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Use muted colors for suspended state
    bg_color = QColor('#45475a') if suspended else QColor('#89b4fa')
    text_color = QColor('#bac2de') if suspended else QColor('#1e1e2e')

    painter.setBrush(bg_color)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(0, 0, size, size, 14, 14)

    painter.setPen(text_color)
    font = QFont('Segoe UI', 24, QFont.Weight.Bold)
    painter.setFont(font)
    painter.drawText(
        0, 0, size, size,
        Qt.AlignmentFlag.AlignCenter,
        'SL'
    )

    if suspended:
        # Draw a small pause indicator
        painter.setBrush(QColor('#f38ba8'))
        margin = 10
        w = 6
        h = 16
        painter.drawRect(size - margin - w*2 - 2, size - margin - h, w, h)
        painter.drawRect(size - margin - w, size - margin - h, w, h)

    painter.end()
    return pixmap


class SystemTrayApp(QSystemTrayIcon):
    """System tray icon with context menu."""
    suspension_signal = pyqtSignal(bool)

    def __init__(self, settings_window, hook_manager, icon_path=None, parent=None):
        self._icon_path = icon_path
        if icon_path and os.path.exists(icon_path):
            icon = QIcon(icon_path)
        else:
            pixmap = _create_tray_icon_pixmap()
            icon = QIcon(pixmap)
        super().__init__(icon, parent)

        self.settings_window = settings_window
        self.hook_manager = hook_manager

        # Thread-safe OSD notification
        from ui.osd import SuspensionOSD
        self.osd = SuspensionOSD()
        self.suspension_signal.connect(self._handle_suspension)

        self.setToolTip('SwitchLang — Keyboard layout auto-switcher')

        self._build_menu()

        self.activated[QSystemTrayIcon.ActivationReason].connect(self._on_activated)

    def _build_menu(self):
        """Build the tray context menu."""
        menu = QMenu()

        title_action = QAction('SwitchLang', menu)
        title_action.triggered.connect(self._show_GITHUB_repo)
        menu.addAction(title_action)

        menu.addSeparator()

        self.toggle_action = QAction('Disable', menu)
        self.toggle_action.triggered.connect(self._toggle_engine)
        menu.addAction(self.toggle_action)

        settings_action = QAction('Settings...', menu)
        settings_action.triggered.connect(self._show_settings)
        menu.addAction(settings_action)

        menu.addSeparator()

        quit_action = QAction('Quit', menu)
        quit_action.triggered.connect(self._quit)
        menu.addAction(quit_action)

        self.setContextMenu(menu)
        self._engine_enabled = True

    def _on_activated(self, reason):
        """Handle tray icon activation (left-click or double-click opens settings)."""
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self._show_settings()

    def _show_settings(self):
        """Show the settings window."""
        self.settings_window.show()
        self.settings_window.activateWindow()
        self.settings_window.raise_()

    def _show_GITHUB_repo(self):
        webbrowser.open('https://github.com/Bumblebee621/SwitchLang')

    def _toggle_engine(self):
        """Toggle the engine on/off from the tray menu."""
        self._engine_enabled = not self._engine_enabled
        self.hook_manager.set_enabled(self._engine_enabled)

        if self._engine_enabled:
            self.toggle_action.setText('Disable')
            self.showMessage(
                'SwitchLang',
                'Auto-switching enabled',
                QSystemTrayIcon.MessageIcon.Information,
                1500
            )
        else:
            self.toggle_action.setText('Enable')
            self.showMessage(
                'SwitchLang',
                'Auto-switching disabled',
                QSystemTrayIcon.MessageIcon.Warning,
                1500
            )

    def notify_suspension(self, suspended):
        """Bridge method called from HookManager (background thread)."""
        self.suspension_signal.emit(suspended)

    @pyqtSlot(bool)
    def _handle_suspension(self, suspended):
        """Actual UI update logic running on the main thread."""
        # Update Tray Icon
        if self._icon_path and os.path.exists(self._icon_path):
            pass
        else:
            self.setIcon(QIcon(_create_tray_icon_pixmap(suspended=suspended)))

        if suspended:
            dur = self.hook_manager._suspend_duration
            self.osd.show_status(
                'SwitchLang Suspended',
                f'Auto-switching paused for {dur} seconds.\n'
                'Press hotkey again to resume.',
                duration_ms=3000
            )
        else:
            self.osd.show_status(
                'SwitchLang Resumed',
                'Auto-switching has resumed.',
                duration_ms=1500
            )

    def _quit(self):
        """Quit the application."""
        self.hook_manager.stop()
        QApplication.instance().quit()

    def update_from_settings(self, config_data):
        """Update tray state when settings change.

        Args:
            config_data: Dict from config.json.
        """
        enabled = config_data.get('enabled', True)
        self._engine_enabled = enabled
        self.hook_manager.set_enabled(enabled)
        self.toggle_action.setText(
            'Disable' if enabled else 'Enable'
        )
