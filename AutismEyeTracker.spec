# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Autism Eye Tracking System

import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# Collect data files from key packages
mediapipe_datas = collect_data_files('mediapipe')
ctk_datas       = collect_data_files('customtkinter')

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=collect_dynamic_libs('mediapipe'),
    datas=[
        # Package data
        *mediapipe_datas,
        *ctk_datas,
        # Project assets
        ('stimuli',       'stimuli'),
        ('app',           'app'),
        ('tracking',      'tracking'),
        ('data',          'data'),
        ('config.py',     '.'),
    ],
    hiddenimports=[
        # Scikit-learn
        'sklearn.ensemble',
        'sklearn.ensemble._forest',
        'sklearn.preprocessing',
        'sklearn.tree',
        'sklearn.tree._classes',
        'sklearn.utils._bunch',
        'sklearn.utils._weight_vector',
        # OpenCV
        'cv2',
        # MediaPipe
        'mediapipe',
        'mediapipe.python.solutions.face_mesh',
        'mediapipe.python.solutions.drawing_utils',
        'mediapipe.python.solutions.drawing_styles',
        # CustomTkinter
        'customtkinter',
        # Matplotlib (required by mediapipe internally)
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends.backend_agg',
        # Other
        'PIL._tkinter_finder',
        'pkg_resources.py2_warn',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['jupyter', 'IPython', 'notebook'],
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
    name='AutismEyeTracker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,      # No black terminal window when launching
    icon='app_icon.ico',
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AutismEyeTracker',
)
