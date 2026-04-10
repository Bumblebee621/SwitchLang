#!/usr/bin/env bash
# ==============================================================================
# SwitchLang Linux Installer
# ==============================================================================
# Usage (one-liner):
#   curl -fsSL https://raw.githubusercontent.com/Bumblebee621/SwitchLang/main/installers/linux/install.sh | bash
#
# What this script does:
#   1. Installs system dependencies (libxcb-cursor0, xdotool) via apt
#   2. Adds your user to the 'input' group for keyboard access
#   3. Downloads the latest SwitchLang binary from GitHub Releases
#   4. Installs it to ~/.local/bin/SwitchLang
#   5. Creates a desktop entry (~/.local/share/applications/SwitchLang.desktop)
#   6. Enables autostart (~/.config/autostart/SwitchLang.desktop)
# ==============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Colour

info()    { echo -e "${BLUE}==>${NC} ${BOLD}$*${NC}"; }
success() { echo -e "${GREEN}  ✔${NC} $*"; }
warn()    { echo -e "${YELLOW}  ⚠${NC} $*"; }
error()   { echo -e "${RED}  ✘${NC} $*" >&2; }
die()     { error "$*"; exit 1; }

# ── Config ────────────────────────────────────────────────────────────────────
REPO="Bumblebee621/SwitchLang"
GITHUB_API="https://api.github.com/repos/${REPO}/releases/latest"
ASSET_NAME="SwitchLang"                          # Linux binary name on GitHub Releases
INSTALL_DIR="${HOME}/.local/bin"
INSTALL_PATH="${INSTALL_DIR}/${ASSET_NAME}"
APP_DESKTOP_DIR="${HOME}/.local/share/applications"
AUTOSTART_DIR="${HOME}/.config/autostart"
DESKTOP_FILENAME="switchlang.desktop"

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}  SwitchLang — Linux Installer${NC}"
echo    "  Real-time keyboard layout auto-switcher (EN ↔ HE)"
echo    "  ─────────────────────────────────────────────────"
echo ""

# ── 0. Preflight checks ───────────────────────────────────────────────────────
info "Checking prerequisites..."

if ! command -v curl &>/dev/null; then
    die "curl is required but not installed. Install it with: sudo apt install curl"
fi

if [[ "$XDG_SESSION_TYPE" == "wayland" ]]; then
    warn "Wayland session detected. SwitchLang currently requires X11."
    warn "If you're running XWayland, the app may still work."
    echo ""
fi

# ── 1. System dependencies ────────────────────────────────────────────────────
info "Installing system dependencies..."

if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y libxcb-cursor0 xdotool
    success "libxcb-cursor0 and xdotool installed."
else
    warn "apt-get not found. Please install manually: libxcb-cursor0, xdotool"
    warn "On Fedora/RHEL: sudo dnf install xcb-util-cursor xdotool"
    warn "On Arch:        sudo pacman -S xcb-util-cursor xdotool"
fi

# ── 2. Input group permissions ────────────────────────────────────────────────
info "Configuring keyboard access permissions..."

if groups "$USER" | grep -q '\binput\b'; then
    success "User '${USER}' is already in the 'input' group."
else
    sudo usermod -aG input "$USER"
    success "Added '${USER}' to the 'input' group."
    echo ""
    warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    warn " You must LOG OUT and LOG BACK IN for this to take effect."
    warn " SwitchLang will not work until you do so."
    warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
fi

# ── 3. Download latest binary ─────────────────────────────────────────────────
info "Fetching latest release info from GitHub..."

RELEASE_JSON=$(curl -fsSL "${GITHUB_API}") \
    || die "Failed to fetch release info from GitHub. Check your internet connection."

DOWNLOAD_URL=$(echo "${RELEASE_JSON}" \
    | grep -o '"browser_download_url": *"[^"]*'"${ASSET_NAME}"'"' \
    | grep -o 'https://[^"]*' \
    | head -1)

VERSION=$(echo "${RELEASE_JSON}" \
    | grep -o '"tag_name": *"[^"]*"' \
    | grep -o '"[^"]*"$' \
    | tr -d '"' \
    | head -1)

if [[ -z "${DOWNLOAD_URL}" ]]; then
    die "Could not find a '${ASSET_NAME}' asset in the latest release (${VERSION:-unknown}).\nMake sure the Linux binary has been uploaded to GitHub Releases."
fi

success "Found release ${VERSION}."
info "Downloading SwitchLang ${VERSION}..."

mkdir -p "${INSTALL_DIR}"
curl -fsSL --progress-bar "${DOWNLOAD_URL}" -o "${INSTALL_PATH}"
chmod +x "${INSTALL_PATH}"
success "Downloaded to ${INSTALL_PATH}."

# ── 4. App menu desktop entry ─────────────────────────────────────────────────
info "Creating application menu entry..."

mkdir -p "${APP_DESKTOP_DIR}"
cat > "${APP_DESKTOP_DIR}/${DESKTOP_FILENAME}" <<EOF
[Desktop Entry]
Name=SwitchLang
Comment=Real-time keyboard layout auto-switcher (EN ↔ HE)
Exec=${INSTALL_PATH}
Icon=input-keyboard
Type=Application
Categories=Utility;Accessibility;
Keywords=keyboard;layout;hebrew;english;switcher;
StartupNotify=false
NoDisplay=false
EOF

success "Desktop entry created at ${APP_DESKTOP_DIR}/${DESKTOP_FILENAME}."

# ── 5. Autostart on login ─────────────────────────────────────────────────────
info "Enabling autostart on login..."

mkdir -p "${AUTOSTART_DIR}"
cat > "${AUTOSTART_DIR}/${DESKTOP_FILENAME}" <<EOF
[Desktop Entry]
Name=SwitchLang
Comment=Real-time keyboard layout auto-switcher (EN ↔ HE)
Exec=${INSTALL_PATH}
Icon=input-keyboard
Type=Application
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
EOF

success "Autostart enabled at ${AUTOSTART_DIR}/${DESKTOP_FILENAME}."

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}  ✔ SwitchLang ${VERSION} installed successfully!${NC}"
echo ""
echo    "  To start now:    ${INSTALL_PATH}"
echo    "  To uninstall:    rm ${INSTALL_PATH} ${APP_DESKTOP_DIR}/${DESKTOP_FILENAME} ${AUTOSTART_DIR}/${DESKTOP_FILENAME}"
echo ""

if groups "$USER" | grep -q '\binput\b' 2>/dev/null; then
    info "You can start SwitchLang now from your application menu or by running:"
    echo "    SwitchLang &"
else
    warn "Remember to log out and log back in before running SwitchLang."
fi

echo ""
