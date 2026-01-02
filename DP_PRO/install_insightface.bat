@echo off
chcp 65001 >nul
echo ========================================
echo InsightFace 安装脚本（使用国内镜像源）
echo ========================================
echo.

echo ⚠️  重要提示：
echo InsightFace 在 Windows 上可能需要 Visual C++ Build Tools（C++桌面开发）。
echo.
echo 您可以选择安装：
echo   - Visual Studio 2026/2022 Community（完整版，推荐）
echo   - Visual C++ Build Tools（仅编译工具）
echo.
set /p continue="是否已安装 Visual Studio 2026/2022 或 Build Tools？(Y/N): "

if /i not "%continue%"=="Y" (
    echo.
    echo 请先安装 Visual C++ Build Tools！
    echo 说明请查看 README.md
    echo.
    pause
    exit /b 1
)

echo.
echo 正在使用清华大学镜像源安装 InsightFace 和 onnxruntime...
echo 这可能需要几分钟时间，请耐心等待...
echo.

python -m pip install insightface onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ 安装成功！
    echo.
    echo 正在验证安装...
    python -c "import insightface; print('InsightFace 导入成功！')"
    
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo ✅ 验证成功！InsightFace 已正确安装。
        echo.
        echo 提示：
        echo 1. 首次运行时会自动下载模型文件（约 200MB）
        echo 2. 如果有 NVIDIA GPU，可以安装 GPU 版本：
        echo    python -m pip install onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
        echo.
    ) else (
        echo.
        echo ⚠️  安装完成，但验证失败。请检查错误信息。
    )
) else (
    echo.
    echo ❌ 安装失败！
    echo.
    echo 可能的原因：
    echo 1. 未安装 Visual C++ Build Tools
    echo 2. 未重新打开命令行窗口（安装 Build Tools 后需要重启）
    echo 3. 网络连接问题
    echo.
    echo 尝试使用其他镜像源...
    echo.
    echo 尝试使用阿里云镜像源...
    python -m pip install insightface onnxruntime -i https://mirrors.aliyun.com/pypi/simple/
    
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo ✅ 使用阿里云镜像源安装成功！
    ) else (
        echo.
        echo ❌ 安装仍然失败。
        echo.
        echo 请检查：
        echo 1. 是否已安装 Visual C++ Build Tools
        echo 2. 是否已重新打开命令行窗口
        echo 3. Python 版本是否为 3.8-3.11
        echo 4. 网络连接是否正常
        echo.
        echo 详细说明请查看: README.md
    )
)

echo.
pause

