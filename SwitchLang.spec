# -*- mode: python ; coding: utf-8 -*-

production_datas = [
    ('data/en_quadgrams.marisa', 'data'),
    ('data/en_quadgrams.meta.json', 'data'),
    ('data/he_quadgrams.marisa', 'data'),
    ('data/he_quadgrams.meta.json', 'data'),
    ('data/so_quadgrams.marisa', 'data'),
    ('data/so_quadgrams.meta.json', 'data'),
    ('data/collisions.json', 'data'),
    ('data/icon.ico', 'data'),
    ('data/icon.png', 'data'),
    ('ui/style.qss', 'ui'),
    ('ui/check.svg', 'ui'),
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=production_datas,
    hiddenimports=['marisa_trie'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SwitchLang',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['data/icon.ico'],
)
