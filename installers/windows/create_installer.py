"""
This script creates an installer for the SwitchLang application.
It uses Inno Setup to create a self-extracting executable that installs the application to the user's home directory.
This script calls build.py to create the executable and then uses Inno Setup to create the installer.

Run from the project root:
    python installers/windows/create_installer.py
"""

import subprocess
import os
import sys

# Resolve project root (two levels up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Path to setup.iss (same directory as this script)
SETUP_ISS = os.path.join(SCRIPT_DIR, 'setup.iss')


def run_command(command, description, cwd=None):
    print(f"--- {description} ---")
    try:
        subprocess.check_call(command, shell=True, cwd=cwd)
        print("Success!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during {description}: {e}\n")
        return False

def create_installer():
    # 0. Sync version from core/version.py to setup.iss
    if not run_command(
        f'python scripts/sync_version.py',
        'Synchronizing version',
        cwd=PROJECT_ROOT
    ):
        print("Aborting: Version sync failed.")
        return

    # 1. Run build.py to create the executable
    if not run_command(
        'python build.py',
        'Building executable with PyInstaller',
        cwd=PROJECT_ROOT
    ):
        print("Aborting: Build failed.")
        return

    # 2. Look for Inno Setup compiler
    iscc_path = r"C:\Program Files (x86)\Inno Setup 6\iscc.exe"

    if not os.path.exists(iscc_path):
        print("--- Inno Setup Compiler Not Found ---")
        print("Please install Inno Setup 6 from https://jrsoftware.org/isdl.php")
        print("After installation, you can run the compiler manually on 'installers/windows/setup.iss'")
        print("or update this script with the correct path to iscc.exe.")
        return

    # 3. Run Inno Setup compiler against setup.iss in this directory
    if run_command(f'"{iscc_path}" "{SETUP_ISS}"', "Compiling Setup Script"):
        print("Installer created successfully in the 'Output' directory!")

if __name__ == "__main__":
    create_installer()
