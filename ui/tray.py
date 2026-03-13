"""
tray.py — System tray icon and context menu.
"""


from PyQt6.QtWidgets import QSystemTrayIcon, QMenu, QApplication
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QAction
from PyQt6.QtCore import Qt, QSize
import os


def _create_tray_icon_pixmap():
    """Create a programmatic 'SL' icon for the system tray."""
    size = 64
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(QColor(0, 0, 0, 0))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    painter.setBrush(QColor('#89b4fa'))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(0, 0, size, size, 14, 14)

    painter.setPen(QColor('#1e1e2e'))
    font = QFont('Segoe UI', 24, QFont.Weight.Bold)
    painter.setFont(font)
    painter.drawText(
        0, 0, size, size,
        Qt.AlignmentFlag.AlignCenter,
        'SL'
    )

    painter.end()
    return pixmap


class SystemTrayApp(QSystemTrayIcon):
    """System tray icon with context menu."""

    def __init__(self, settings_window, hook_manager, icon_path=None, parent=None):
        if icon_path and os.path.exists(icon_path):
            icon = QIcon(icon_path)
        else:
            pixmap = _create_tray_icon_pixmap()
            icon = QIcon(pixmap)
        super().__init__(icon, parent)

        self.settings_window = settings_window
        self.hook_manager = hook_manager

        self.setToolTip('SwitchLang — Keyboard layout auto-switcher')

        self._build_menu()

        self.activated[QSystemTrayIcon.ActivationReason].connect(self._on_activated)

    def _build_menu(self):
        """Build the tray context menu."""
        menu = QMenu()

        title_action = QAction('SwitchLang', menu)
        title_action.setEnabled(False)
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
