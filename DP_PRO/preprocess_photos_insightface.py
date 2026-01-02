"""
学生照片预处理脚本（使用 InsightFace）
将照片中的人脸居中裁剪为统一尺寸，统一格式为 JPG
"""
import cv2
import os
import numpy as np
from pathlib import Path
import sys
from typing import Optional, Tuple
import csv

# 尝试导入 InsightFace
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("错误: InsightFace 未安装")
    print("请运行: python -m pip install insightface onnxruntime")
    sys.exit(1)

# onnxruntime providers（用于启用GPU）
try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None

def _select_onnx_providers():
    force_cpu = os.environ.get("INSIGHTFACE_FORCE_CPU", "").strip() in {"1", "true", "True", "YES", "yes"}
    if force_cpu:
        return ["CPUExecutionProvider"]
    if ort is not None:
        try:
            avail = set(ort.get_available_providers())
            # 默认不尝试 TensorRT（避免缺少 TensorRT DLL 时刷屏 error 126）
            use_trt = os.environ.get("INSIGHTFACE_USE_TENSORRT", "").strip() in {"1", "true", "True", "YES", "yes"}
            providers = []
            if "CUDAExecutionProvider" in avail:
                providers.append("CUDAExecutionProvider")
            if use_trt and ("TensorrtExecutionProvider" in avail):
                providers.insert(0, "TensorrtExecutionProvider")
            if "DmlExecutionProvider" in avail:
                providers.append("DmlExecutionProvider")
            providers.append("CPUExecutionProvider")
            seen = set()
            providers = [p for p in providers if not (p in seen or seen.add(p))]
            return providers
        except Exception:
            pass
    return ["CPUExecutionProvider"]


def _is_under_root(p: Path, root: Path) -> bool:
    try:
        p_rel = p.resolve()
        r_rel = root.resolve()
        return str(p_rel).lower().startswith(str(r_rel).lower() + os.sep) or str(p_rel).lower() == str(r_rel).lower()
    except Exception:
        return False


def _safe_resolve_dir(user_path: str, root: Path) -> Path:
    """
    安全策略：不要直接用 sys.argv 作为任意文件路径。
    这里做基本校验：必须是目录，且必须位于项目根目录内。
    """
    p = Path(user_path).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()
    if not p.exists() or not p.is_dir():
        raise ValueError(f"路径不是有效目录: {p}")
    if not _is_under_root(p, root):
        raise ValueError(f"出于安全考虑，目录必须位于项目根目录内: {p}")
    return p


def _safe_resolve_file(user_path: str, root: Path) -> Path:
    """
    安全策略：不要直接用 sys.argv 作为任意文件路径。
    这里做基本校验：必须是文件，且必须位于项目根目录内。
    """
    p = Path(user_path).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()
    if not p.exists() or not p.is_file():
        raise ValueError(f"路径不是有效文件: {p}")
    if not _is_under_root(p, root):
        raise ValueError(f"出于安全考虑，文件必须位于项目根目录内: {p}")
    return p


def _load_student_list_csv(csv_path: Path) -> Tuple[dict, dict]:
    """
    读取 student_list.csv，返回：
    - name_to_id: 姓名 -> 学号
    - id_to_name: 学号 -> 姓名
    """
    name_to_id = {}
    id_to_name = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return ({}, {})
        # 兼容列名（学号/姓名 或 student_id/name）
        cols = [c.strip() for c in reader.fieldnames]
        id_col = None
        name_col = None
        for c in cols:
            cl = c.lower()
            if ("学号" in c) or ("student_id" in cl) or (cl == "id"):
                id_col = c
            if ("姓名" in c) or ("name" in cl):
                name_col = c
        if id_col is None or name_col is None:
            # 回退：取前两列
            id_col = cols[0]
            name_col = cols[1] if len(cols) > 1 else cols[0]
        for row in reader:
            sid = str((row.get(id_col) or "")).strip()
            name = str((row.get(name_col) or "")).strip()
            if sid.endswith(".0"):
                sid = sid[:-2]
            if sid and name:
                name_to_id[name] = sid
                id_to_name[sid] = name
    return (name_to_id, id_to_name)


def _safe_imread(path: Path) -> Optional[np.ndarray]:
    """
    支持 Windows 中文路径的读图方式：np.fromfile + cv2.imdecode。
    避免 cv2.imread 在某些环境下对非ASCII路径失败。
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data is None or data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _safe_imwrite_jpg(path: Path, img_bgr: np.ndarray, quality: int = 95) -> bool:
    """
    支持 Windows 中文路径的写图方式：cv2.imencode + Python 写文件。
    避免 cv2.imwrite 在某些环境下对非ASCII路径写出“乱码文件名”或失败。
    """
    try:
        q = int(quality)
        if q < 30:
            q = 30
        if q > 100:
            q = 100
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            return False
        # 用 Python 文件IO写入，路径支持Unicode
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False


def preprocess_student_photos(
    input_dir: str, 
    output_dir: str = "student_photos_processed", 
    target_size: tuple = (512, 512),
    padding_ratio: float = 1.5,
    student_list_csv: Optional[str] = None,
):
    """
    预处理学生照片，将人脸居中裁剪为统一尺寸
    
    Args:
        input_dir: 输入照片目录
        output_dir: 输出照片目录
        target_size: 目标尺寸 (width, height)，默认 (512, 512)
        padding_ratio: 裁剪区域相对于人脸大小的倍数（默认1.5，即包含一些背景）
    """
    print("="*60)
    print("学生照片预处理工具（InsightFace 版本）")
    print("="*60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"目标尺寸: {target_size[0]}x{target_size[1]}")
    print(f"裁剪比例: {padding_ratio}x")
    if student_list_csv:
        print(f"名单映射: {student_list_csv}（将输出命名为 学号.jpg）")
    print()

    name_to_id = {}
    if student_list_csv:
        try:
            name_to_id, _ = _load_student_list_csv(Path(student_list_csv))
            print(f"已加载名单映射: {len(name_to_id)} 条")
        except Exception as e:
            print(f"⚠️  警告: 读取名单CSV失败，将继续使用原文件名输出。原因: {e}")
            name_to_id = {}
    
    # 初始化 InsightFace
    print("正在初始化 InsightFace...")
    providers = _select_onnx_providers()
    if providers and providers[0] in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}:
        print(f"✅ 预处理将优先使用 GPU（{providers[0]}）")
    elif providers and providers[0] == "DmlExecutionProvider":
        print("✅ 预处理将优先使用 GPU（DmlExecutionProvider/DirectML）")
    else:
        print("ℹ️  预处理将使用 CPU（CPUExecutionProvider）")
    try:
        face_app = FaceAnalysis(
            name='buffalo_l',
            providers=providers
        )
        face_app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as e:
        print("⚠️  警告: 预处理尝试启用 GPU 失败，将自动回退。")
        print(f"    失败原因: {e}")
        fallback = ["DmlExecutionProvider", "CPUExecutionProvider"] if ("DmlExecutionProvider" in (providers or [])) else ["CPUExecutionProvider"]
        if fallback[0] == "DmlExecutionProvider":
            print("ℹ️  回退到 DirectML（DmlExecutionProvider）继续使用 GPU")
        else:
            print("ℹ️  回退到 CPU（CPUExecutionProvider）")
        face_app = FaceAnalysis(name='buffalo_l', providers=fallback)
        face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("InsightFace 初始化完成")
    print()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有照片文件
    photo_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        photo_files.extend(Path(input_dir).glob(ext))
        photo_files.extend(Path(input_dir).glob(f'**/{ext}'))  # 递归搜索
    
    # 去重
    photo_files = list(set([str(p.resolve()) for p in photo_files]))
    photo_files = [Path(p) for p in photo_files]
    
    print(f"找到 {len(photo_files)} 个照片文件")
    print("开始处理...")
    print()
    
    success_count = 0
    failed_count = 0
    
    for idx, photo_path in enumerate(photo_files, 1):
        try:
            # 读取图像（BGR格式）
            img = _safe_imread(photo_path)
            if img is None:
                print(f"  [{idx}/{len(photo_files)}] 警告: {photo_path.name} - 无法读取图像")
                failed_count += 1
                continue
            
            # 使用 InsightFace 检测人脸
            faces = face_app.get(img)
            
            if len(faces) == 0:
                print(f"  [{idx}/{len(photo_files)}] 警告: {photo_path.name} - 未检测到人脸，跳过")
                failed_count += 1
                continue
            
            # 使用第一张检测到的人脸
            face = faces[0]
            bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            
            # 计算人脸中心
            face_center_x = (x1 + x2) // 2
            face_center_y = (y1 + y2) // 2
            
            # 计算人脸尺寸
            face_width = x2 - x1
            face_height = y2 - y1
            
            # 计算裁剪区域大小（以人脸为中心，包含一些背景）
            crop_size = int(max(face_width, face_height) * padding_ratio)
            
            # 获取图像尺寸
            h, w = img.shape[:2]
            
            # 确保裁剪尺寸不超过图像大小
            crop_size = min(crop_size, w, h)
            
            # 计算裁剪区域的左上角坐标
            crop_x = max(0, face_center_x - crop_size // 2)
            crop_y = max(0, face_center_y - crop_size // 2)
            
            # 确保不超出图像边界
            if crop_x + crop_size > w:
                crop_x = w - crop_size
            if crop_y + crop_size > h:
                crop_y = h - crop_size
            
            # 确保坐标不为负
            crop_x = max(0, crop_x)
            crop_y = max(0, crop_y)
            
            # 裁剪图像
            cropped = img[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
            
            # 如果裁剪区域为空，跳过
            if cropped.size == 0:
                print(f"  [{idx}/{len(photo_files)}] 警告: {photo_path.name} - 裁剪区域无效")
                failed_count += 1
                continue
            
            # 调整到目标尺寸
            resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)
            
            # 保存处理后的图像（统一为 JPG 格式）
            stem = os.path.splitext(photo_path.name)[0]
            out_stem = stem
            # 优先：若本身就是学号文件名，保持不变
            if not stem.isdigit():
                # 尝试用姓名映射到学号
                mapped = name_to_id.get(stem) if name_to_id else None
                if mapped:
                    out_stem = mapped
            output_filename = out_stem + '.jpg'
            output_path = Path(output_dir) / output_filename
            # 保存为 JPG，质量设置为 95（使用安全写入，避免中文文件名乱码）
            if not _safe_imwrite_jpg(output_path, resized, quality=95):
                print(f"  [{idx}/{len(photo_files)}] 警告: {photo_path.name} - 写入失败")
                failed_count += 1
                continue
            
            success_count += 1
            if success_count <= 5 or success_count % 10 == 0 or idx == len(photo_files):
                print(f"  [{idx}/{len(photo_files)}] 已处理: {photo_path.name} -> {output_filename}")
        
        except Exception as e:
            print(f"  [{idx}/{len(photo_files)}] 错误: 处理 {photo_path.name} 时出错: {e}")
            failed_count += 1
    
    print()
    print("="*60)
    print("处理完成！")
    print("="*60)
    print(f"成功: {success_count} 个")
    print(f"失败: {failed_count} 个")
    print(f"处理后的照片保存在: {output_dir}/")
    print()
    print("现在可以在运行分析时使用预处理后的照片目录:")
    print(f"  python main.py --video ... --photos {output_dir} --list student_list.csv --task 1")
    print()


if __name__ == "__main__":
    project_root = Path(os.getcwd()).resolve()
    input_dir = "student_photos"
    output_dir = "student_photos_processed"
    list_csv = None
    
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    if len(sys.argv) > 3:
        list_csv = sys.argv[3]
    
    try:
        in_dir = _safe_resolve_dir(input_dir, project_root)
        # output_dir 允许不存在，但必须在项目根目录下
        out_dir = Path(output_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = (project_root / out_dir).resolve()
        else:
            out_dir = out_dir.resolve()
        if not _is_under_root(out_dir, project_root):
            raise ValueError(f"出于安全考虑，输出目录必须位于项目根目录内: {out_dir}")
        csv_path = None
        if list_csv:
            csv_path = str(_safe_resolve_file(list_csv, project_root))
        preprocess_student_photos(str(in_dir), str(out_dir), student_list_csv=csv_path)
    except Exception as e:
        print(f"错误: {e}")
        print("用法: python preprocess_photos_insightface.py [输入目录] [输出目录]")
        print("示例: python preprocess_photos_insightface.py \"学生照片目录\" student_photos_processed student_list.csv")
        sys.exit(1)

