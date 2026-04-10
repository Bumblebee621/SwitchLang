import PyInstaller.__main__
import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# PyInstaller uses ';' as the path separator on Windows and ':' on Linux/macOS
_SEP = ';' if sys.platform == 'win32' else ':'
_ICON = 'data/icon.ico' if sys.platform == 'win32' else 'data/icon.png'

PyInstaller.__main__.run([
    'main.py',
    '--name=SwitchLang',
    '--noconsole',
    '--onefile',
    f'--icon={_ICON}',
    f'--add-data=data{_SEP}data',
    f'--add-data=ui{_SEP}ui',
    '--clean'
])
