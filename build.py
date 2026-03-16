import PyInstaller.__main__
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))

PyInstaller.__main__.run([
    'main.py',
    '--name=SwitchLang',
    '--noconsole',
    '--onefile',
    '--icon=data/icon.ico',
    '--add-data=data;data',
    '--add-data=ui;ui',
    '--clean'
])
