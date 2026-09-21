"""
updater.py — Logic for checking and downloading updates from GitHub.
"""

import json
import os
import subprocess
import tempfile
import urllib.request
from core import __version__

REPO = "Bumblebee621/SwitchLang"
GITHUB_API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
_USER_AGENT = "SwitchLang-Updater"

def check_for_updates():
    """
    Checks GitHub for a newer version.
    Returns: (new_version_string, download_url) if higher version exists, else (None, None).
    """
    try:
        req = urllib.request.Request(GITHUB_API_URL, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        latest_tag = data.get("tag_name", "").lstrip("v")
        if not latest_tag:
            return None, None
            
        if _is_version_higher(latest_tag, __version__):
            # Look for SwitchLang_Setup.exe in assets
            for asset in data.get("assets", []):
                if asset["name"] == "SwitchLang_Setup.exe":
                    return latest_tag, asset["browser_download_url"]
            
            # Fallback to the first asset if specifically named one not found
            if data.get("assets"):
                return latest_tag, data["assets"][0]["browser_download_url"]
                
        return None, None
    except Exception as e:
        print(f"Error checking for updates: {e}")
        return None, None

def _is_version_higher(latest, current):
    """Simple semantic version comparison."""
    try:
        l_parts = [int(p) for p in latest.split(".")]
        c_parts = [int(p) for p in current.split(".")]
        
        # Pad with zeros if necessary
        max_len = max(len(l_parts), len(c_parts))
        l_parts.extend([0] * (max_len - len(l_parts)))
        c_parts.extend([0] * (max_len - len(c_parts)))
        
        return l_parts > c_parts
    except (ValueError, AttributeError):
        return False

def download_and_install(url, progress_callback=None):
    """
    Downloads the installer and runs it.
    progress_callback: function(current_bytes, total_bytes)
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        temp_dir = tempfile.gettempdir()
        installer_path = os.path.join(temp_dir, "SwitchLang_Setup.exe")

        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = int(response.getheader('Content-Length', 0))
            downloaded = 0
            with open(installer_path, "wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size)
                        
        # Use os._exit(0) instead of sys.exit(0) to release exe file lock immediately before installer runs.
        import time
        subprocess.Popen([installer_path, "/SILENT"])
        time.sleep(1)
        os._exit(0)
    except Exception as e:
        print(f"Error downloading update: {e}")
        return False
