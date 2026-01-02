@echo off
chcp 65001 >nul
echo ========================================
echo 环境准备脚本（适用于从其他电脑复制的项目）
echo ========================================
echo.

echo 步骤1: 检查 Python 环境...
echo.

REM 检查 python 命令
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Python 已安装
    python --version
    goto :check_pip
)

REM 检查 py 命令
py --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Python 已安装（通过 py 命令）
    py --version
    set PYTHON_CMD=py
    goto :check_pip
)

echo ❌ 未找到 Python！
echo.
echo 请先安装 Python 3.8-3.11:
echo 1. 访问: https://www.python.org/downloads/
echo 2. 下载并安装 Python 3.9 或 3.10
echo 3. 安装时务必勾选 "Add Python to PATH"
echo 4. 安装完成后，重新打开命令行窗口
echo.
pause
exit /b 1

:check_pip
echo.
echo 步骤2: 检查 pip...
echo.

if defined PYTHON_CMD (
    %PYTHON_CMD% -m pip --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ pip 已安装
        %PYTHON_CMD% -m pip --version
        set PIP_CMD=%PYTHON_CMD% -m pip
        goto :upgrade_pip
    )
) else (
    python -m pip --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ pip 已安装
        python -m pip --version
        set PIP_CMD=python -m pip
        goto :upgrade_pip
    )
)

echo ❌ pip 未安装或无法使用
echo.
pause
exit /b 1

:upgrade_pip
echo.
echo 步骤3: 升级 pip...
echo.
%PIP_CMD% install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo 步骤4: 检查已安装的包...
echo.

if defined PYTHON_CMD (
    %PYTHON_CMD% -c "import cv2; print('✅ OpenCV 已安装')" 2>nul || echo ❌ OpenCV 未安装
    %PYTHON_CMD% -c "import numpy; print('✅ NumPy 已安装')" 2>nul || echo ❌ NumPy 未安装
    %PYTHON_CMD% -c "import pandas; print('✅ Pandas 已安装')" 2>nul || echo ❌ Pandas 未安装
    %PYTHON_CMD% -c "import insightface; print('✅ InsightFace 已安装')" 2>nul || echo ❌ InsightFace 未安装
    %PYTHON_CMD% -c "import ultralytics; print('✅ Ultralytics 已安装')" 2>nul || echo ❌ Ultralytics 未安装
) else (
    python -c "import cv2; print('✅ OpenCV 已安装')" 2>nul || echo ❌ OpenCV 未安装
    python -c "import numpy; print('✅ NumPy 已安装')" 2>nul || echo ❌ NumPy 未安装
    python -c "import pandas; print('✅ Pandas 已安装')" 2>nul || echo ❌ Pandas 未安装
    python -c "import insightface; print('✅ InsightFace 已安装')" 2>nul || echo ❌ InsightFace 未安装
    python -c "import ultralytics; print('✅ Ultralytics 已安装')" 2>nul || echo ❌ Ultralytics 未安装
)

echo.
echo ========================================
echo 步骤5: 安装依赖包
echo ========================================
echo.
echo 本项目任务1/2需要安装：
echo 1. requirements.txt（OpenCV/NumPy/Pandas/Ultralytics/ONNX 等）
echo 2. InsightFace（任务1必需，可能需要 Visual C++ Build Tools）
echo.
echo 注意：如果 InsightFace 安装失败，通常是缺少 Visual C++ Build Tools 或未重开命令行
echo.

set /p install_now="是否现在安装所有依赖包？(Y/N): "
if /i not "%install_now%"=="Y" (
    echo.
    echo 已跳过安装。您可以稍后运行：
    echo   install_dependencies.bat  - 安装基础依赖
    echo   install_insightface.bat   - 安装 InsightFace
    echo.
    pause
    exit /b 0
)

echo.
echo 正在安装基础依赖包...
echo.

REM 安装 requirements.txt
%PIP_CMD% install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

if %errorlevel% neq 0 (
    echo.
    echo ⚠️  requirements.txt 安装失败，建议先修复后再继续
)

echo.
echo ========================================
echo 安装 InsightFace（需要 Visual C++ Build Tools）
echo ========================================
echo.
echo 如果您的电脑已安装 Visual Studio 2026/2022 或 Build Tools，可以继续安装 InsightFace
echo.

set /p install_insightface="是否安装 InsightFace？(Y/N): "
if /i "%install_insightface%"=="Y" (
    echo.
    echo 正在安装 InsightFace 和 onnxruntime...
    %PIP_CMD% install insightface onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
    
    if %errorlevel% equ 0 (
        echo.
        echo ✅ InsightFace 安装成功！
        echo.
        echo 正在验证...
        if defined PYTHON_CMD (
            %PYTHON_CMD% -c "import insightface; print('✅ InsightFace 验证成功！')" 2>nul
        ) else (
            python -c "import insightface; print('✅ InsightFace 验证成功！')" 2>nul
        )
    ) else (
        echo.
        echo ❌ InsightFace 安装失败！
        echo.
        echo 可能的原因：
        echo 1. 未安装 Visual C++ Build Tools
        echo 2. 未重新打开命令行窗口（安装 Build Tools 后需要重启）
        echo.
        echo 解决方案：
echo 1. 安装 Visual Studio 2022/2026 或 Build Tools（C++桌面开发）
        echo 3. 重新打开命令行窗口
        echo 4. 再次运行此脚本或 install_insightface.bat
    )
) else (
    echo.
    echo 已跳过 InsightFace 安装
    echo 您可以稍后运行 install_insightface.bat 来安装
)

echo.
echo ========================================
echo 环境准备完成！
echo ========================================
echo.
echo 提示：
echo 1. 任务1需要 InsightFace；任务2需要 Ultralytics（会自动下载 .pt 权重）
echo 2. 首次运行会自动下载模型文件（可能需要网络）
echo 3. 详细说明请查看 README.md
echo.
pause

