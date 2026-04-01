"""
blacklist.py — Foreground application polling and dynamic blacklist.

Detects the currently focused application (via the platform backend)
and checks it against a user-managed blacklist of process names.
"""

import json
import logging
import os

logger = logging.getLogger('switchlang.blacklist')

DEFAULT_BLACKLIST = {
    'keepass.exe', 'keepassxc.exe', '1password.exe',
    'bitwarden.exe', 'credentialuibroker.exe', 'consent.exe',
    # Linux equivalents (matched without .exe by convention)
    'keepass', 'keepassxc', '1password', 'bitwarden',
}

IDE_EXECUTABLES = {
    # JetBrains (Windows)
    'pycharm64.exe', 'pycharm.exe', 'idea64.exe', 'idea.exe',
    'webstorm64.exe', 'webstorm.exe', 'clion64.exe', 'clion.exe',
    'datagrip64.exe', 'datagrip.exe', 'rider64.exe', 'rider.exe',
    'rubymine64.exe', 'rubymine.exe', 'goland64.exe', 'goland.exe',
    'phpstorm64.exe', 'phpstorm.exe', 'studio64.exe', 'studio.exe',
    # JetBrains (Linux — process names without .exe)
    'pycharm', 'idea', 'webstorm', 'clion', 'datagrip', 'rider',
    'rubymine', 'goland', 'phpstorm', 'android-studio',
    # Microsoft
    'code.exe', 'insiders.exe', 'devenv.exe',
    'code', 'code-insiders',  # Linux
    # Editors
    'vim.exe', 'gvim.exe', 'nvim.exe', 'nvim-qt.exe',
    'vim', 'gvim', 'nvim', 'nvim-qt',  # Linux
    # Others
    'eclipse.exe', 'netbeans64.exe', 'netbeans.exe',
    'codeblocks.exe', 'qtcreator.exe', 'spyder.exe',
    'rstudio.exe', 'matlab.exe',
    'eclipse', 'netbeans', 'codeblocks', 'qtcreator', 'spyder',
    'rstudio', 'matlab',  # Linux
}


class BlacklistManager:
    """Manages the set of blacklisted process names.

    Process names are stored without OS-specific extensions. On Windows,
    the full .exe name is used (e.g. 'code.exe'). On Linux, the bare
    process name is used (e.g. 'code'). The comparison is case-insensitive.
    """

    def __init__(self, config_path):
        """Initialize from config.json.

        Args:
            config_path: Path to the config.json file.
        """
        self.config_path = config_path
        self.blacklisted = set()
        self.tech_apps = set()
        self._platform = None
        self._load()

    def set_platform(self, platform):
        """Set the platform backend for foreground process detection.

        Args:
            platform: PlatformBackend instance.
        """
        self._platform = platform

    def get_foreground_exe(self):
        """Get the process name of the currently focused window.

        Delegates to the platform backend.

        Returns:
            Process name (e.g. 'code.exe' or 'code'), or '' on failure.
        """
        if self._platform is None:
            return ''
        return self._platform.get_foreground_process()

    def _load(self):
        """Load blacklist from config file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                user_blacklist = data.get('blacklist', [])
                self.blacklisted = set(exe.lower() for exe in user_blacklist)
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
            exe_name: The process name (e.g. 'code.exe' or 'code').
        """
        self.blacklisted.add(exe_name.lower())
        self.save()

    def remove(self, exe_name):
        """Remove an executable from the blacklist.

        Args:
            exe_name: The process name to remove.
        """
        self.blacklisted.discard(exe_name.lower())
        self.save()

    def is_ide_editor(self, exe_name=None):
        """Check if a process is an IDE or code editor.

        Args:
            exe_name: Process name to check.

        Returns:
            True if the process is a recognized IDE/editor or custom technical app.
        """
        if exe_name is None:
            return False
        exe_lower = exe_name.lower()
        return exe_lower in IDE_EXECUTABLES or exe_lower in self.tech_apps

    def get_list(self):
        """Get the sorted list of blacklisted executables.

        Returns:
            Sorted list of process names.
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
