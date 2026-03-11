import os
import sys
import winreg
import logging

logger = logging.getLogger('switchlang.startup')

def get_current_app_command():
    """Returns the command line string to run the current application.
    
    If frozen (EXE), returns the absolute path to the EXE.
    If script, returns 'python.exe "path/to/main.py"'.
    """
    if getattr(sys, 'frozen', False):
        return f'"{sys.executable}"'
    else:
        # For development, we want to run main.py with the current python interpreter
        python_exe = sys.executable
        script_path = os.path.abspath(sys.argv[0])
        return f'"{python_exe}" "{script_path}"'

def is_startup_enabled(app_name="SwitchLang"):
    """Check if the application is set to run on Windows startup."""
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ) as key:
            try:
                value, _ = winreg.QueryValueEx(key, app_name)
                # Check if the registered path matches our current command (roughly)
                current_cmd = get_current_app_command()
                # Use in string check because paths might have quotes or relative bits
                return current_cmd.lower() in value.lower() or value.lower() in current_cmd.lower()
            except FileNotFoundError:
                return False
    except Exception as e:
        logger.error(f"Error checking startup registry: {e}")
        return False

def set_startup_enabled(enabled, app_name="SwitchLang"):
    """Enable or disable application launch on Windows startup."""
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE) as key:
            if enabled:
                cmd = get_current_app_command()
                winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, cmd)
                logger.info(f"Enabled startup: {cmd}")
            else:
                try:
                    winreg.DeleteValue(key, app_name)
                    logger.info("Disabled startup.")
                except FileNotFoundError:
                    pass
        return True
    except Exception as e:
        logger.error(f"Error modifying startup registry: {e}")
        return False
