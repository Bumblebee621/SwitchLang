import PyInstaller.__main__
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))

PyInstaller.__main__.run([
    'main.py',
    '--name=SwitchLang',
    '--noconsole',
    '--onefile',
    f'--icon={os.path.join(APP_DIR, "data", "icon.ico")}',
    f'--add-data={os.path.join(APP_DIR, "data")};data',
    f'--add-data={os.path.join(APP_DIR, "ui")};ui',
    '--clean'
])
