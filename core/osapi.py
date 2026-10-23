"""osapi.py — the OS input layer for this platform (see os_windows / os_macos / os_linux)."""

import sys

if sys.platform == 'win32':
    from core.os_windows import *  # noqa: F401,F403
elif sys.platform == 'darwin':
    from core.os_macos import *  # noqa: F401,F403
else:
    from core.os_linux import *  # noqa: F401,F403
