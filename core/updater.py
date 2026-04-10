"""
updater.py — Logic for checking and downloading updates from GitHub.
"""

import requests
import os
import sys
import subprocess
import tempfile
from core.version import __version__

REPO = "Bumblebee621/SwitchLang"
GITHUB_API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"

# Asset name varies by platform
_ASSET_NAME = "SwitchLang_Setup.exe" if sys.platform == "win32" else "SwitchLang"

def check_for_updates():
    """
    Checks GitHub for a newer version.
    Returns: (new_version_string, download_url) if higher version exists, else (None, None).
    """
    try:
        response = requests.get(GITHUB_API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        latest_tag = data.get("tag_name", "").lstrip("v")
        if not latest_tag:
            return None, None
            
        if _is_version_higher(latest_tag, __version__):
            # Look for the platform-appropriate asset
            for asset in data.get("assets", []):
                if asset["name"] == _ASSET_NAME:
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
    Downloads the update and installs it.
    On Windows: runs the Inno Setup installer silently.
    On Linux: replaces the current binary in-place and relaunches.
    progress_callback: function(current_bytes, total_bytes)
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        temp_dir = tempfile.gettempdir()
        installer_path = os.path.join(temp_dir, _ASSET_NAME)
        
        downloaded = 0
        with open(installer_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size)

        if sys.platform == "win32":
            # Launch the Inno Setup installer silently, then exit
            # Using os._exit(0) instead of sys.exit(0) to ensure the process
            # is fully terminated and the exe file is released before the
            # installer tries to replace it.
            import time
            subprocess.Popen([installer_path, "/SILENT"])
            time.sleep(1)
            os._exit(0)
        else:
            # Linux: replace the running binary with the downloaded one,
            # then relaunch from the new version.
            import stat
            os.chmod(installer_path, os.stat(installer_path).st_mode | stat.S_IEXEC)
            current_binary = sys.executable if getattr(sys, 'frozen', False) else None
            if current_binary:
                import shutil
                shutil.move(installer_path, current_binary)
                subprocess.Popen([current_binary])
            else:
                subprocess.Popen([installer_path])
            import time
            time.sleep(1)
            os._exit(0)

    except Exception as e:
        print(f"Error downloading update: {e}")
        return False

