"""
blacklist.py — Foreground application polling and dynamic blacklist.

Uses the OS layer (core.osapi) to detect the currently focused application
and check it against a user-managed blacklist of executables.
"""

import json
import logging
import os

from core import osapi

logger = logging.getLogger('switchlang.blacklist')

DEFAULT_BLACKLIST = {
    'keepass.exe', 'keepassxc.exe', '1password.exe',
    'bitwarden.exe', 'credentialuibroker.exe', 'consent.exe',
    # Linux (as /proc/<pid>/comm reports them)
    'keepassxc', '1password', 'bitwarden'
}

IDE_EXECUTABLES = {
    # JetBrains
    'pycharm64.exe', 'pycharm.exe', 'idea64.exe', 'idea.exe',
    'webstorm64.exe', 'webstorm.exe', 'clion64.exe', 'clion.exe',
    'datagrip64.exe', 'datagrip.exe', 'rider64.exe', 'rider.exe',
    'rubymine64.exe', 'rubymine.exe', 'goland64.exe', 'goland.exe',
    'phpstorm64.exe', 'phpstorm.exe', 'studio64.exe', 'studio.exe',
    # Microsoft
    'code.exe', 'insiders.exe', 'devenv.exe',
    # Editors
    'vim.exe', 'gvim.exe', 'nvim.exe', 'nvim-qt.exe',
    # Linux (as /proc/<pid>/comm reports them)
    'code', 'codium', 'pycharm', 'idea', 'webstorm', 'clion', 'goland', 'rider',
    'studio', 'nvim-qt', 'gvim', 'zed', 'kate',
    # Others
    'eclipse.exe', 'netbeans64.exe', 'netbeans.exe',
    'codeblocks.exe', 'qtcreator.exe', 'spyder.exe',
    'rstudio.exe', 'matlab.exe'
}


class BlacklistManager:
    """Manages the set of blacklisted exe names."""

    def __init__(self, config_path):
        """Initialize from config.json.

        Args:
            config_path: Path to the config.json file.
        """
        self.config_path = config_path
        self.blacklisted = set()
        self.tech_apps = set()
        self._load()

    def _load(self):
        """Load blacklist from config file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                user_blacklist = data.get('blacklist', [])
                self.blacklisted = set(exe.lower() for exe in user_blacklist)
                # If first time or empty, ensure defaults are added
                if not self.blacklisted and not user_blacklist:
                    self.blacklisted.update(DEFAULT_BLACKLIST)
                
                user_tech_apps = data.get('tech_apps', [])
                self.tech_apps = set(exe.lower() for exe in user_tech_apps)
            except json.JSONDecodeError:
                logger.error('Malformed config file: %s — using default blacklist',
                             self.config_path)
                self.blacklisted = set(DEFAULT_BLACKLIST)
                self.tech_apps = set()
        else:
            self.blacklisted = set(DEFAULT_BLACKLIST)
            self.tech_apps = set()

    def save(self):
        """Persist the current blacklist to config.json."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                logger.error('Malformed config file: %s — overwriting with blacklist only',
                             self.config_path)
                data = {}
        else:
            data = {}

        data['blacklist'] = sorted(self.blacklisted)
        data['tech_apps'] = sorted(self.tech_apps)

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
        return osapi.get_foreground_exe()

    def is_blacklisted(self):
        """Check if the current foreground app is blacklisted.

        Returns:
            True if the current foreground exe is in the blacklist.
        """
        exe = self.get_foreground_exe()
        result = exe in self.blacklisted
        if result:
            logger.debug('Blacklisted app active: %s', exe)
        return result

    def is_ide_editor(self, exe_name=None):
        """Check if a process is an IDE or code editor.

        Args:
            exe_name: Optional exe name to check. If None, checks foreground app.

        Returns:
            True if the process is a recognized IDE/editor or custom technical app.
        """
        if exe_name is None:
            exe_name = self.get_foreground_exe()
        exe_lower = exe_name.lower()
        return exe_lower in IDE_EXECUTABLES or exe_lower in self.tech_apps

    def get_list(self):
        """Get the sorted list of blacklisted executables.

        Returns:
            Sorted list of exe names.
        """
        return sorted(self.blacklisted)

    def add_tech_app(self, exe_name):
        """Add an executable to the technical apps list."""
        self.tech_apps.add(exe_name.lower())
        self.save()

    def remove_tech_app(self, exe_name):
        """Remove an executable from the technical apps list."""
        self.tech_apps.discard(exe_name.lower())
        self.save()

    def get_tech_apps_list(self):
        """Get the sorted list of technical apps."""
        return sorted(self.tech_apps)
