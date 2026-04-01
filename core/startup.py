"""
startup.py — Startup management (cross-platform).

Delegates to the platform backend for OS-specific autostart logic:
- Windows: Registry (HKCU\\...\\Run)
- Linux: XDG autostart (~/.config/autostart/*.desktop)
"""

import logging

logger = logging.getLogger('switchlang.startup')

# Platform backend instance — set by main.py at startup
_platform = None


def set_platform(platform):
    """Set the platform backend for startup operations.

    Args:
        platform: PlatformBackend instance.
    """
    global _platform
    _platform = platform


def is_startup_enabled(app_name="SwitchLang"):
    """Check if the application is set to run on OS startup."""
    if _platform is None:
        logger.warning('Platform not set — cannot check startup status')
        return False
    return _platform.is_startup_enabled(app_name)


def set_startup_enabled(enabled, app_name="SwitchLang"):
    """Enable or disable application launch on OS startup."""
    if _platform is None:
        logger.warning('Platform not set — cannot modify startup')
        return False
    return _platform.set_startup_enabled(enabled, app_name)
