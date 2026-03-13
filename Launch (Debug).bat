@echo off
cd /d "%~dp0"
echo Starting SwitchLang...
.venv\Scripts\python.exe main.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo Application crashed with error code %ERRORLEVEL%
    pause
)
