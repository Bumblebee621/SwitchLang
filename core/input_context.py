"""A window/control identity and the keyboard layout observed with it."""
from dataclasses import dataclass


@dataclass(frozen=True)
class InputContext:
    window: object
    focus: object
    layout: str

    @property
    def identity(self):
        return self.window, self.focus


def read_context(get_identity, get_layout):
    """Reject a layout sample if focus moved while the query was in flight."""
    try:
        window, focus = get_identity()
        layout = get_layout()
        if not window or (window, focus) != get_identity():
            return None
        return InputContext(window, focus, layout)
    except Exception:
        return None
