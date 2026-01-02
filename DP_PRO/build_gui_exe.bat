@echo off
setlocal enabledelayedexpansion

cd /d %~dp0

REM Windows CMD 默认不是 UTF-8，切换到 UTF-8 避免中文乱码
chcp 65001 >nul

REM 使用清华源（避免联网慢/超时）
set PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
set PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
echo [INFO] pip index: %PIP_INDEX_URL%
set ORT_VER=1.19.2
echo [INFO] ORT_VER=%ORT_VER% (CPU)

REM 关闭可能正在运行的旧版 EXE，避免 PyInstaller 删除 dist 时 WinError 5（拒绝访问）
taskkill /F /IM DP_PRO_GUI.exe >nul 2>nul

REM 清理旧产物（如果资源管理器正在打开 dist\DP_PRO_GUI 也可能锁文件；建议关掉窗口再运行）
if exist "dist\DP_PRO_GUI" (
  rmdir /S /Q "dist\DP_PRO_GUI" >nul 2>nul
)
if exist "build\dp_pro_gui" (
  rmdir /S /Q "build\dp_pro_gui" >nul 2>nul
)

echo ============================================================
echo [1/3] 安装依赖（建议在虚拟环境 venv 中执行）
echo ============================================================
python -m pip install -U pip -i %PIP_INDEX_URL% --trusted-host %PIP_TRUSTED_HOST%
python -m pip install -r requirements.txt -i %PIP_INDEX_URL% --trusted-host %PIP_TRUSTED_HOST%

REM 关键：强制安装 CPU 版 onnxruntime（并用 --no-deps，避免把 numpy/protobuf/sympy 升级到不兼容版本）
python -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml >nul 2>nul
echo [INFO] 安装 onnxruntime==%ORT_VER%（CPU）
python -m pip install --upgrade --force-reinstall onnxruntime==%ORT_VER% --no-deps -i %PIP_INDEX_URL% --trusted-host %PIP_TRUSTED_HOST%

REM 兜底：强制把关键版本锁回项目要求，避免上一步/历史环境污染导致冲突
python -m pip install --upgrade --force-reinstall numpy==1.24.3 protobuf==3.20.3 sympy==1.13.1 --no-deps -i %PIP_INDEX_URL% --trusted-host %PIP_TRUSTED_HOST%

REM 防止 InsightFace->albumentations 依赖链把 opencv-python-headless 装回来，导致 OpenCV GUI 功能缺失
REM （即便 GUI 本身用 Qt，不用 cv2 弹窗；但你后续可能还要用 --t2-interactive-roi）
python -m pip uninstall -y opencv-python-headless opencv-contrib-python-headless >nul 2>nul
python -m pip install --upgrade --force-reinstall opencv-python==4.8.1.78 opencv-contrib-python==4.8.1.78 --no-deps -i %PIP_INDEX_URL% --trusted-host %PIP_TRUSTED_HOST%

echo.
echo ============================================================
echo [2/3] 准备离线模型 assets/（首次可能需要联网下载）
echo ============================================================
python prepare_offline_assets.py

echo.
echo ============================================================
echo [3/3] PyInstaller 打包（输出在 dist\DP_PRO_GUI\）
echo ============================================================
python -m PyInstaller -y dp_pro_gui.spec

echo.
echo [OK] 完成：dist\DP_PRO_GUI\DP_PRO_GUI.exe
pause


