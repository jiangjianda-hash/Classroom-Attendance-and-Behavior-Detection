@echo off
chcp 65001 >nul
echo ========================================
echo 依赖安装脚本（任务1/任务2）
echo ========================================
echo.
echo 注意：本项目使用 InsightFace（任务1必需），任务2使用 Ultralytics（YOLO/RT-DETR）
echo.

echo 步骤1: 安装 requirements.txt（推荐固定版本，避免 numpy/protobuf 冲突）...
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

if %errorlevel% neq 0 (
    echo.
    echo 错误: requirements.txt 安装失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo 依赖安装完成！
echo ========================================
echo.
echo 提示：
echo 1. 若 InsightFace 安装失败：请先安装 Visual C++ Build Tools，然后运行 install_insightface.bat
echo 2. 若任务2弹窗框选 ROI 失败（cvNamedWindow）：请确认未安装 opencv-python-headless
echo.
echo 验证安装:
python -c "import cv2; print('✅ OpenCV 版本:', cv2.__version__)" 2>nul || echo ❌ OpenCV 未安装
python -c "import numpy; print('✅ NumPy 版本:', numpy.__version__)" 2>nul || echo ❌ NumPy 未安装
python -c "import pandas; print('✅ Pandas 版本:', pandas.__version__)" 2>nul || echo ❌ Pandas 未安装
python -c "import insightface; print('✅ InsightFace 已安装')" 2>nul || echo ⚠️ InsightFace 未安装（任务1需要）
python -c "import ultralytics; print('✅ Ultralytics 已安装')" 2>nul || echo ⚠️ Ultralytics 未安装（任务2需要）

echo.
pause

