@echo off
chcp 65001 >nul
echo ========================================
echo 环境完整性检查
echo ========================================
echo.

REM 检测 Python 命令
set PYTHON_CMD=python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    py --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=py
    ) else (
        echo ❌ 未找到 Python！
        pause
        exit /b 1
    )
)

echo [1/8] 检查 Python 版本...
%PYTHON_CMD% --version
for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python 版本检查完成
echo.

echo [2/8] 检查核心依赖包...
echo.

set MISSING_PACKAGES=0

%PYTHON_CMD% -c "import cv2; print('✅ OpenCV', cv2.__version__)" 2>nul || (echo ❌ OpenCV 未安装 & set /a MISSING_PACKAGES+=1)
%PYTHON_CMD% -c "import numpy; print('✅ NumPy', numpy.__version__)" 2>nul || (echo ❌ NumPy 未安装 & set /a MISSING_PACKAGES+=1)
%PYTHON_CMD% -c "import pandas; print('✅ Pandas', pandas.__version__)" 2>nul || (echo ❌ Pandas 未安装 & set /a MISSING_PACKAGES+=1)
%PYTHON_CMD% -c "import PIL; print('✅ Pillow', PIL.__version__)" 2>nul || (echo ❌ Pillow 未安装 & set /a MISSING_PACKAGES+=1)
%PYTHON_CMD% -c "import sklearn; print('✅ scikit-learn', sklearn.__version__)" 2>nul || (echo ❌ scikit-learn 未安装 & set /a MISSING_PACKAGES+=1)
%PYTHON_CMD% -c "import openpyxl; print('✅ openpyxl', openpyxl.__version__)" 2>nul || (echo ❌ openpyxl 未安装 & set /a MISSING_PACKAGES+=1)
%PYTHON_CMD% -c "import google.protobuf; print('✅ protobuf 已安装')" 2>nul || (echo ❌ protobuf 未安装 & set /a MISSING_PACKAGES+=1)
%PYTHON_CMD% -c "import onnx; print('✅ onnx 已安装')" 2>nul || (echo ❌ onnx 未安装 & set /a MISSING_PACKAGES+=1)
%PYTHON_CMD% -c "import ultralytics; print('✅ ultralytics 已安装')" 2>nul || (echo ❌ ultralytics 未安装 & set /a MISSING_PACKAGES+=1)

echo.
echo [3/8] 检查 InsightFace...
%PYTHON_CMD% -c "import insightface; print('✅ InsightFace 已安装')" 2>nul || (echo ❌ InsightFace 未安装 & set /a MISSING_PACKAGES+=1)

echo.
echo [4/8] 检查 onnxruntime...
%PYTHON_CMD% -c "import onnxruntime; print('✅ onnxruntime', onnxruntime.__version__)" 2>nul || (echo ❌ onnxruntime 未安装 & set /a MISSING_PACKAGES+=1)

echo.
echo [5/8] 检查项目文件...
echo.

if exist "main.py" (
    echo ✅ main.py 存在
) else (
    echo ❌ main.py 不存在
    set /a MISSING_PACKAGES+=1
)

if exist "face_recognition_module.py" (
    echo ✅ face_recognition_module.py 存在
) else (
    echo ❌ face_recognition_module.py 不存在
    set /a MISSING_PACKAGES+=1
)

if exist "task2_behavior_flow.py" (
    echo ✅ task2_behavior_flow.py 存在
) else (
    echo ❌ task2_behavior_flow.py 不存在
    set /a MISSING_PACKAGES+=1
)

if exist "behavior_rules_yolo.py" (
    echo ✅ behavior_rules_yolo.py 存在
) else (
    echo ❌ behavior_rules_yolo.py 不存在
    set /a MISSING_PACKAGES+=1
)

if exist "attendance_checker.py" (
    echo ✅ attendance_checker.py 存在
) else (
    echo ❌ attendance_checker.py 不存在
    set /a MISSING_PACKAGES+=1
)

echo.
echo [6/8] 检查数据文件...
echo.

if exist "student_list.csv" (
    echo ✅ student_list.csv 存在
) else (
    echo ⚠️  student_list.csv 不存在（需要准备数据）
)

if exist "student_photos_processed" (
    echo ✅ student_photos_processed 目录存在
) else (
    echo ⚠️  student_photos_processed 目录不存在（需要准备学生照片，推荐先跑 preprocess_photos_insightface.py）
)

if exist "教室视频" (
    echo ✅ 教室视频 目录存在
) else (
    echo ⚠️  教室视频 目录不存在（需要准备视频文件）
)

echo.
echo [7/8] 测试 InsightFace 模型加载...
echo.

%PYTHON_CMD% -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=0); print('✅ InsightFace 模型加载成功')" 2>nul || (
    echo ⚠️  InsightFace 模型加载失败（首次运行会自动下载模型，约 200MB）
    echo    如果持续失败，请检查网络连接或手动下载模型
)

echo.
echo [8/8] 检查模块导入...
echo.

%PYTHON_CMD% -c "from face_recognition_module import FaceRecognitionModule; print('✅ face_recognition_module 导入成功')" 2>nul || (
    echo ❌ face_recognition_module 导入失败
    set /a MISSING_PACKAGES+=1
)

%PYTHON_CMD% -c "import task2_behavior_flow; print('✅ task2_behavior_flow 导入成功')" 2>nul || (
    echo ❌ task2_behavior_flow 导入失败
    set /a MISSING_PACKAGES+=1
)

%PYTHON_CMD% -c "from attendance_checker import AttendanceChecker; print('✅ attendance_checker 导入成功')" 2>nul || (
    echo ❌ attendance_checker 导入失败
    set /a MISSING_PACKAGES+=1
)

echo.
echo ========================================
echo 检查结果汇总
echo ========================================
echo.

if %MISSING_PACKAGES% equ 0 (
    echo ✅ 所有核心依赖已安装！
    echo.
    echo 项目已准备就绪，可以开始使用：
    echo.
    echo 示例命令：
    echo   python main.py --video "教室视频/1105.mp4" --photos student_photos_processed --list student_list.csv --task 1
    echo   python main.py --video "教室视频/1105.mp4" --task 2 --t2-mode person --t2-interactive-roi --t2-save-marked-images
    echo.
) else (
    echo ⚠️  发现 %MISSING_PACKAGES% 个问题需要解决
    echo.
    echo 建议操作：
    echo 1. 如果缺少依赖包，运行: setup_environment.bat
    echo 2. 如果缺少数据文件，运行: prepare_data.py
    echo 3. 如果 InsightFace 模型加载失败，检查网络连接
    echo.
)

echo.
pause

