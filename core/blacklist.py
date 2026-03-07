"""
blacklist.py — Foreground application polling and dynamic blacklist.

Uses Windows APIs to detect the currently focused application and
check it against a user-managed blacklist of executables.
"""

import ctypes
import ctypes.wintypes as wintypes
import os
import json

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class BlacklistManager:
    """Manages the set of blacklisted exe names."""

    def __init__(self, config_path):
        """Initialize from config.json.

        Args:
            config_path: Path to the config.json file.
        """
        self.config_path = config_path
        self.blacklisted = set()
        self._load()

    def _load(self):
        """Load blacklist from config file."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.blacklisted = set(
                exe.lower() for exe in data.get('blacklist', [])
            )

    def save(self):
        """Persist the current blacklist to config.json."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {}

        data['blacklist'] = sorted(self.blacklisted)

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def add(self, exe_name):
        """Add an executable to the blacklist.

        Args:
            exe_name: The .exe filename (e.g. 'code.exe').
        """
        self.blacklisted.add(exe_name.lower())
        self.save()

    def remove(self, exe_name):
        """Remove an executable from the blacklist.

        Args:
            exe_name: The .exe filename to remove.
        """
        self.blacklisted.discard(exe_name.lower())
        self.save()

    def get_foreground_exe(self):
        """Get the executable name of the currently focused window.

        Returns:
            The .exe filename (e.g. 'notepad.exe'), or '' on failure.
        """
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ''

        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

        if pid.value == 0:
            return ''

        h_process = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value
        )
        if not h_process:
            return ''

        try:
            buf = ctypes.create_unicode_buffer(512)
            size = wintypes.DWORD(512)
            success = kernel32.QueryFullProcessImageNameW(
                h_process, 0, buf, ctypes.byref(size)
            )
            if success:
                full_path = buf.value
                return os.path.basename(full_path).lower()
            return ''
        finally:
            kernel32.CloseHandle(h_process)

    def is_blacklisted(self):
        """Check if the current foreground app is blacklisted.

        Returns:
            True if the current foreground exe is in the blacklist.
        """
        exe = self.get_foreground_exe()
        return exe in self.blacklisted

    def get_list(self):
        """Get the sorted list of blacklisted executables.

        Returns:
            Sorted list of exe names.
        """
        return sorted(self.blacklisted)
