# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# PyInstaller hooks: 收集 onnxruntime 的 provider DLL（CPU/CUDA/DirectML 都在这里面）
from PyInstaller.utils.hooks import collect_dynamic_libs

# 说明：
# 不要 collect_submodules("torch")：torch 子模块极多，且不同版本差异大，会导致海量 “hidden import not found”、
# 构建时间暴涨。PyInstaller 自带的 hook-torch/hook-torchvision 已能覆盖绝大多数运行需求。
hiddenimports = [
    "ultralytics",
    "cv2",
    "onnxruntime",
    "insightface",
    # 本项目本地模块：有些是在函数内动态 import，PyInstaller 静态分析可能漏掉
    "face_recognition_module",
    "attendance_checker",
    "task2_behavior_flow",
    "behavior_rules_yolo",
    "preprocess_photos_insightface",
    "prepare_data",
    "app_paths",
]

datas = [
    ("assets", "assets"),
    ("README.md", "."),
    # 本地单文件模块：作为 data 直接拷贝到 _internal 根目录，确保运行时 import 可用
    ("face_recognition_module.py", "."),
    ("attendance_checker.py", "."),
    ("task2_behavior_flow.py", "."),
    ("behavior_rules_yolo.py", "."),
    ("preprocess_photos_insightface.py", "."),
    ("prepare_data.py", "."),
    ("main.py", "."),
    ("app_paths.py", "."),
]

# 关键：把 onnxruntime 的动态库（尤其是 providers DLL）显式打进包里
binaries = collect_dynamic_libs("onnxruntime")

a = Analysis(
    ["gui_app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    name="DP_PRO_GUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DP_PRO_GUI",
)


