# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['nadi_main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'numpy',
        'scipy',
        'scipy.signal',
        'scipy.integrate',
        'PyQt6',
        'pyqtgraph',
        'pyqtgraph.opengl',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AyurvedicNadiPariksha',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AyurvedicNadiPariksha',
)
