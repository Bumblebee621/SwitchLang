{
  lib,
  stdenvNoCC,
  python3,
  qt6,
  makeWrapper,
  glib,
  dconf,
  gsettings-desktop-schemas,
  libX11,
  setxkbmap,
}:
let
  runtimePackages = ps: [
    ps.pyqt6
    ps.marisa-trie
  ];
  python = python3.withPackages runtimePackages;
  testPython = python3.withPackages (ps: runtimePackages ps ++ [ ps.pytest ]);
in
stdenvNoCC.mkDerivation {
  pname = "switchlang";
  version = builtins.head (
    builtins.match ".*__version__ = \"([^\"]+)\".*" (builtins.readFile ../core/__init__.py)
  );

  # Only runtime sources and tests: never copy corpora, logs or local builds.
  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../main.py
      (lib.fileset.fileFilter (file: file.hasExt "py") ../core)
      (lib.fileset.fileFilter (
        file:
        builtins.any file.hasExt [
          "py"
          "qss"
          "svg"
        ]
      ) ../ui)
      ../data/en_quadgrams.marisa
      ../data/en_quadgrams.meta.json
      ../data/he_quadgrams.marisa
      ../data/he_quadgrams.meta.json
      ../data/so_quadgrams.marisa
      ../data/so_quadgrams.meta.json
      ../data/collisions.json
      ../data/icon.png
      ../installers/linux
      ../tests
      (lib.fileset.fileFilter (file: file.hasExt "py") ../evaluation)
      ../LICENSE
      ../THIRD_PARTY_LICENSES.md
    ];
  };

  nativeBuildInputs = [
    makeWrapper
    qt6.wrapQtAppsHook
  ];
  buildInputs = [
    qt6.qtbase
    qt6.qtwayland
  ];
  dontWrapQtApps = true;
  dontBuild = true;

  postPatch = ''
    substituteInPlace core/desktop_x11.py \
      --replace-fail "ctypes.util.find_library('X11') or 'libX11.so.6'" "'${lib.getLib libX11}/lib/libX11.so.6'"
  '';

  doCheck = true;
  checkPhase = ''
    runHook preCheck
    export PYTHONDONTWRITEBYTECODE=1
    ${testPython}/bin/python -m pytest tests -q
    runHook postCheck
  '';

  installPhase = ''
    runHook preInstall
    app="$out/share/switchlang"
    mkdir -p "$app" "$out/bin" "$out/share/applications" "$out/share/pixmaps" "$out/lib/udev/rules.d"
    cp main.py "$app/"
    cp -r core ui data "$app/"
    install -m644 LICENSE THIRD_PARTY_LICENSES.md "$app/"
    install -m644 data/icon.png "$out/share/pixmaps/switchlang.png"
    install -m644 installers/linux/70-switchlang.rules "$out/lib/udev/rules.d/"
    substitute installers/linux/switchlang.desktop "$out/share/applications/switchlang.desktop" \
      --replace-fail '@EXEC@' "$out/bin/switchlang" \
      --replace-fail '@ICON@' 'switchlang'
    makeWrapper ${python}/bin/python "$out/bin/switchlang" \
      --add-flags "$app/main.py" \
      --set SWITCHLANG_EXECUTABLE switchlang \
      --prefix PATH : ${
        lib.makeBinPath [
          glib
          setxkbmap
        ]
      } \
      --prefix XDG_DATA_DIRS : ${gsettings-desktop-schemas}/share/gsettings-schemas/${gsettings-desktop-schemas.name} \
      --prefix GIO_EXTRA_MODULES : ${lib.getLib dconf}/lib/gio/modules \
      "''${qtWrapperArgs[@]}"
    runHook postInstall
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    # Exercise the installed wrapper and Qt plugins without a desktop or devices.
    export XDG_CONFIG_HOME="$TMPDIR/switchlang-config"
    export XDG_RUNTIME_DIR="$TMPDIR/switchlang-runtime"
    mkdir -p "$XDG_CONFIG_HOME" "$XDG_RUNTIME_DIR"
    chmod 700 "$XDG_RUNTIME_DIR"
    # The startup error dialog waits for interaction because no devices or desktop
    # exist in the sandbox. An early exit instead indicates a packaging failure.
    status=0
    env -u DISPLAY -u WAYLAND_DISPLAY -u HYPRLAND_INSTANCE_SIGNATURE \
      XDG_CURRENT_DESKTOP= QT_QPA_PLATFORM=offscreen \
      timeout 5 "$out/bin/switchlang" || status=$?
    test "$status" -eq 124
    runHook postInstallCheck
  '';

  meta = {
    description = "Real-time English/Hebrew keyboard layout auto-switcher";
    homepage = "https://github.com/Bumblebee621/SwitchLang";
    license = lib.licenses.gpl3Only;
    mainProgram = "switchlang";
    platforms = [
      "x86_64-linux"
      "aarch64-linux"
    ];
  };
}
