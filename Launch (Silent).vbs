Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\Ariel\Documents\Antigravity\SwitchLang"
WshShell.Run ".venv\Scripts\pythonw.exe main.py", 0, False
