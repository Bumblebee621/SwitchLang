#!/usr/bin/env bash
# SwitchLang installer for Linux. Run it from the extracted release folder:
#   ./install.sh              install for the current user
#   ./install.sh --uninstall  remove it again
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
app="$HOME/.local/share/switchlang"
menu="$HOME/.local/share/applications/switchlang.desktop"
autostart="$HOME/.config/autostart/switchlang.desktop"  # same file Settings → "Run on startup" toggles
rule=/etc/udev/rules.d/70-switchlang.rules

pkill -x SwitchLang || true  # stop a running copy (upgrade or uninstall)

if [ "${1:-}" = --uninstall ]; then
    rm -rf "$app" "$menu" "$autostart" "$HOME/.local/bin/switchlang"
    sudo rm -f "$rule"
    sudo udevadm control --reload
    echo "SwitchLang removed."
    exit 0
fi

mkdir -p "$app" "$HOME/.local/bin" "$(dirname "$menu")" "$(dirname "$autostart")"
install -m 755 "$here/SwitchLang" "$app/SwitchLang"
install -m 644 "$here/icon.png" "$app/icon.png"
ln -sf "$app/SwitchLang" "$HOME/.local/bin/switchlang"
sed -e "s|@EXEC@|$app/SwitchLang|" -e "s|@ICON@|$app/icon.png|" "$here/switchlang.desktop" > "$menu"
cp "$menu" "$autostart"

echo "SwitchLang reads the keyboard directly, so it needs one system rule (asks for your password)."
sudo install -m 644 "$here/70-switchlang.rules" "$rule"
sudo udevadm control --reload
sudo udevadm trigger --subsystem-match=input --subsystem-match=misc

nohup "$app/SwitchLang" >/dev/null 2>&1 &
echo "SwitchLang is installed, running (tray icon), and starts when you log in."
echo "If it says it can't access the keyboard, log out and back in once."
