"""
环境完整性检查脚本
"""
import sys
import os
import io

# 设置输出编码为 UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def check_python_version():
    """检查 Python 版本"""
    print("=" * 50)
    print("[1/8] 检查 Python 版本...")
    print(f"[OK] Python {sys.version}")
    version_info = sys.version_info
    if version_info.major == 3 and 8 <= version_info.minor <= 11:
        print(f"[OK] Python 版本符合要求 (3.{version_info.minor})")
    else:
        print(f"[WARN] Python 版本 {version_info.major}.{version_info.minor}，推荐使用 3.8-3.11")
    print()

def check_package(package_name, import_name=None, version_attr="__version__"):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        if hasattr(module, version_attr):
            version = getattr(module, version_attr)
            print(f"[OK] {package_name} {version}")
            return True
        else:
            print(f"[OK] {package_name} 已安装")
            return True
    except ImportError:
        print(f"[FAIL] {package_name} 未安装")
        return False

def check_core_dependencies():
    """检查核心依赖包"""
    print("[2/8] 检查核心依赖包...")
    print()
    
    packages = [
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("Pillow", "PIL"),
        ("scikit-learn", "sklearn"),
        ("openpyxl", "openpyxl"),
        ("protobuf", "google.protobuf"),
        ("onnx", "onnx"),
        ("ultralytics", "ultralytics"),
    ]
    
    missing = 0
    for name, import_name in packages:
        if not check_package(name, import_name):
            missing += 1
    
    print()
    return missing

def check_insightface():
    """检查 InsightFace"""
    print("[3/8] 检查 InsightFace...")
    result = check_package("InsightFace", "insightface")
    print()
    return 0 if result else 1

def check_onnxruntime():
    """检查 onnxruntime"""
    print("[4/8] 检查 onnxruntime...")
    result = check_package("onnxruntime", "onnxruntime")
    print()
    return 0 if result else 1

def check_project_files():
    """检查项目文件"""
    print("[5/8] 检查项目文件...")
    print()
    
    required_files = [
        "main.py",
        "face_recognition_module.py",
        "attendance_checker.py",
        "task2_behavior_flow.py",
        "behavior_rules_yolo.py",
    ]
    
    missing = 0
    for file in required_files:
        if os.path.exists(file):
            print(f"[OK] {file} 存在")
        else:
            print(f"[FAIL] {file} 不存在")
            missing += 1
    
    print()
    return missing

def check_data_files():
    """检查数据文件"""
    print("[6/8] 检查数据文件...")
    print()
    
    data_items = [
        ("student_list.csv", "选课名单"),
        ("student_photos_processed", "学生照片目录（预处理后，推荐）"),
        ("教室视频", "教室视频目录"),
    ]
    
    for item, desc in data_items:
        if os.path.exists(item):
            print(f"[OK] {desc} ({item}) 存在")
        else:
            print(f"[WARN] {desc} ({item}) 不存在（需要准备数据）")
    
    print()

def test_insightface_model():
    """测试 InsightFace 模型加载"""
    print("[7/8] 测试 InsightFace 模型加载...")
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0)
        print("[OK] InsightFace 模型加载成功")
        print()
        return True
    except Exception as e:
        print(f"[WARN] InsightFace 模型加载失败: {e}")
        print("   首次运行时会自动下载模型（约 200MB）")
        print("   如果持续失败，请检查网络连接或手动下载模型")
        print()
        return False

def test_module_imports():
    """测试模块导入"""
    print("[8/8] 检查模块导入...")
    print()
    
    modules = [
        ("face_recognition_module", "FaceRecognitionModule"),
        ("attendance_checker", "AttendanceChecker"),
        ("task2_behavior_flow", "run_task2_flow"),
    ]
    
    missing = 0
    for module_name, class_name in modules:
        try:
            module = __import__(module_name)
            getattr(module, class_name)
            print(f"[OK] {module_name} 导入成功")
        except Exception as e:
            print(f"[FAIL] {module_name} 导入失败: {e}")
            missing += 1
    
    print()
    return missing

def main():
    """主函数"""
    print("=" * 50)
    print("环境完整性检查")
    print("=" * 50)
    print()
    
    total_missing = 0
    
    # 检查 Python 版本
    check_python_version()
    
    # 检查核心依赖
    total_missing += check_core_dependencies()
    
    # 检查 InsightFace
    total_missing += check_insightface()
    
    # 检查 onnxruntime
    total_missing += check_onnxruntime()
    
    # 检查项目文件
    total_missing += check_project_files()
    
    # 检查数据文件
    check_data_files()
    
    # 测试 InsightFace 模型
    test_insightface_model()
    
    # 测试模块导入
    total_missing += test_module_imports()
    
    # 汇总结果
    print("=" * 50)
    print("检查结果汇总")
    print("=" * 50)
    print()
    
    if total_missing == 0:
        print("[OK] 所有核心依赖已安装！")
        print()
        print("项目已准备就绪，可以开始使用：")
        print()
        print("示例命令：")
        print('  python main.py --video "教室视频/1105.mp4" --photos student_photos_processed --list student_list.csv --task 1')
        print('  python main.py --video "教室视频/1105.mp4" --task 2 --t2-mode person --t2-interactive-roi --t2-save-marked-images')
        print()
    else:
        print(f"[WARN] 发现 {total_missing} 个问题需要解决")
        print()
        print("建议操作：")
        print("1. 如果缺少依赖包，运行: setup_environment.bat")
        print("2. 如果缺少数据文件，运行: prepare_data.py")
        print("3. 如果 InsightFace 模型加载失败，检查网络连接")
        print()

if __name__ == "__main__":
    main()

