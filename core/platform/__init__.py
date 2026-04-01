"""
platform — OS-specific backend selection.

Auto-detects the current operating system and display server,
then returns the appropriate PlatformBackend implementation.
"""

import sys


def get_platform_backend():
    """Create and return the correct platform backend for the current OS.

    Returns:
        A PlatformBackend instance for the detected platform.

    Raises:
        RuntimeError: If the current platform is not supported.
    """
    if sys.platform == 'win32':
        from core.platform.windows import WindowsBackend
        return WindowsBackend()
    elif sys.platform == 'linux':
        # Future: detect Wayland vs X11
        # if os.environ.get('WAYLAND_DISPLAY'):
        #     from core.platform.linux_wayland import LinuxWaylandBackend
        #     return LinuxWaylandBackend()
        from core.platform.linux_x11 import LinuxX11Backend
        return LinuxX11Backend()
    else:
        raise RuntimeError(f'Unsupported platform: {sys.platform}')
