"""
osd.py — On-Screen Display (OSD) for real-time application feedback.
"""

from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QGraphicsOpacityEffect
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint
from PyQt6.QtGui import QFont, QColor, QScreen, QGuiApplication

class SuspensionOSD(QWidget):
    """A sleek, animated overlay to show engine suspension state."""

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Frameless, transparent, doesn't steal focus, always on top
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        # Layout & Content
        layout = QVBoxLayout(self)
        self.container = QWidget()
        self.container.setObjectName('osd_container')
        # We manually style this in code for maximum portability/WOW factor
        self.container.setStyleSheet("""
            QWidget#osd_container {
                background-color: rgba(30, 30, 46, 230);
                border: 1px solid rgba(137, 180, 250, 100);
                border-radius: 12px;
            }
        """)
        
        inner_layout = QVBoxLayout(self.container)
        inner_layout.setContentsMargins(20, 10, 20, 10)
        inner_layout.setSpacing(2)

        self.title_label = QLabel("SwitchLang")
        self.title_label.setStyleSheet("color: #89b4fa; font-weight: bold; font-size: 14px;")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.status_label = QLabel("Suspended")
        self.status_label.setStyleSheet("color: #cdd6f4; font-size: 12px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        inner_layout.addWidget(self.title_label)
        inner_layout.addWidget(self.status_label)
        layout.addWidget(self.container)

        # Opacity animation for fade in/out
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.opacity_anim.setDuration(300)
        self.opacity_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self.fade_out)

    def show_status(self, title, message, duration_ms=1500):
        """Display the OSD with given text for a specific duration."""
        self.title_label.setText(title)
        self.status_label.setText(message)
        
        # Adjust size based on content
        self.adjustSize()
        hint = self.sizeHint()
        
        # Position at bottom-right of the primary screen
        screen = QGuiApplication.primaryScreen().availableGeometry()
        margin = 20
        x = screen.width() - hint.width() - margin
        y = screen.height() - hint.height() - margin
        self.move(x, y)

        self.show()
        
        # Reset animation state
        self.opacity_anim.stop()
        try:
            self.opacity_anim.finished.disconnect()
        except TypeError:
            pass # No connections to disconnect
            
        self.opacity_anim.setStartValue(self.opacity_effect.opacity())
        self.opacity_anim.setEndValue(1.0)
        self.opacity_anim.start()
        
        self.hide_timer.start(duration_ms)

    def fade_out(self):
        """Smoothly hide the OSD."""
        self.opacity_anim.stop()
        try:
            self.opacity_anim.finished.disconnect()
        except TypeError:
            pass

        self.opacity_anim.setStartValue(self.opacity_effect.opacity())
        self.opacity_anim.setEndValue(0.0)
        self.opacity_anim.finished.connect(self.hide)
        self.opacity_anim.start()

# Helper instance check (Singleton style for app-wide use)
_instance = None

def show_osd(title, message, duration=1500):
    global _instance
    if _instance is None:
        _instance = SuspensionOSD()
    _instance.show_status(title, message, duration)
