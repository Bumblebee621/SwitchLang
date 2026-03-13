"""
This script creates an installer for the SwitchLang application.
It uses Inno Setup to create a self-extracting executable that installs the application to the user's home directory.
This script calls build.py to create the executable and then uses Inno Setup to create the installer.
"""


import subprocess
import os
import sys

def run_command(command, description):
    print(f"--- {description} ---")
    try:
        subprocess.check_call(command, shell=True)
        print("Success!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during {description}: {e}\n")
        return False

def create_installer():
    # 1. Run build.py to create the executable
    if not run_command("python build.py", "Building executable with PyInstaller"):
        print("Aborting: Build failed.")
        return

    # 2. Look for Inno Setup compiler
    iscc_path = r"C:\Program Files (x86)\Inno Setup 6\iscc.exe"
    
    if not os.path.exists(iscc_path):
        print("--- Inno Setup Compiler Not Found ---")
        print("Please install Inno Setup 6 from https://jrsoftware.org/isdl.php")
        print("After installation, you can run the compiler manually on 'setup.iss'")
        print("or update this script with the correct path to iscc.exe.")
        return

    # 3. Run Inno Setup compiler
    print("--- Creating Installer with Inno Setup ---")
    if run_command(f'"{iscc_path}" setup.iss', "Compiling Setup Script"):
        print("Installer created successfully in the 'Output' directory!")

if __name__ == "__main__":
    create_installer()
