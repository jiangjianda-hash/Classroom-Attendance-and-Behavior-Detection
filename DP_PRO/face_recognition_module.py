"""
人脸识别模块
用于从视频中识别学生并匹配选课名单
仅支持 InsightFace（本项目已不再使用 face_recognition/dlib）
"""
import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
import sys
import ctypes


def _maybe_add_cuda_paths_windows():
    """
    Windows 上 onnxruntime CUDA provider (onnxruntime_providers_cuda.dll) 依赖 CUDA/cuDNN DLL。
    这些 DLL 通常不在 Python 包内，而在 CUDA Toolkit 安装目录的 bin 里。
    为了避免“之前能跑、环境变动后突然 126”的情况，这里做一次“仅当前进程”的 PATH 兜底：
    - 只尝试固定的常见 CUDA 安装路径（不使用任何用户输入，符合安全规则）
    - 仅在路径存在且不在 PATH 时追加
    """
    if sys.platform != "win32":
        return
    candidates = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
    ]
    cur = os.environ.get("PATH", "")
    for p in candidates:
        try:
            if os.path.exists(p) and (p not in cur):
                os.environ["PATH"] = p + os.pathsep + cur
                cur = os.environ["PATH"]
        except Exception:
            continue

    # 额外兜底：很多用户安装了 CUDA 版 PyTorch，其自带的 cuDNN DLL 位于 torch\lib。
    # onnxruntime CUDA provider 在加载时不会主动去找 torch 目录，所以这里把 torch\lib
    # 加入“当前进程”的 DLL 搜索路径，避免 126。
    try:
        import torch  # type: ignore

        torch_dir = os.path.dirname(getattr(torch, "__file__", "") or "")
        torch_lib = os.path.join(torch_dir, "lib")
        if os.path.isdir(torch_lib):
            if torch_lib not in cur:
                os.environ["PATH"] = torch_lib + os.pathsep + cur
                cur = os.environ["PATH"]
            # Python 3.8+：为当前进程添加 DLL 搜索目录（比单纯 PATH 更可靠）
            try:
                os.add_dll_directory(torch_lib)  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass

# InsightFace 延迟导入：
# 注意：InsightFace 在部分版本中会间接依赖 albumentations（进而可能拉入 opencv-python-headless）。
# 我们在运行时再尝试导入，并在失败时给出“不会破坏任务2 ROI 弹窗”的安装指引。
FaceAnalysis = None  # type: ignore
INSIGHTFACE_AVAILABLE = False
_INSIGHTFACE_IMPORT_ERR: Optional[str] = None


def _ensure_insightface() -> None:
    """确保 FaceAnalysis 可用；失败则记录错误原因。"""
    global FaceAnalysis, INSIGHTFACE_AVAILABLE, _INSIGHTFACE_IMPORT_ERR
    if INSIGHTFACE_AVAILABLE and FaceAnalysis is not None:
        return
    try:
        # 使用官方入口导入；若缺少 albumentations/qudida 等，会在这里抛出异常
        from insightface.app import FaceAnalysis as _FaceAnalysis  # type: ignore

        FaceAnalysis = _FaceAnalysis  # type: ignore
        INSIGHTFACE_AVAILABLE = True
        _INSIGHTFACE_IMPORT_ERR = None
    except Exception as e:
        FaceAnalysis = None  # type: ignore
        INSIGHTFACE_AVAILABLE = False
        _INSIGHTFACE_IMPORT_ERR = str(e)

# onnxruntime providers（用于启用GPU）
try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None

# 让“导入模块”本身保持可用；真正使用时（初始化）再抛出更友好的错误

# 在 import 时做一次 PATH 兜底：保证后续创建 onnxruntime CUDA session 时更容易找到依赖 DLL
_maybe_add_cuda_paths_windows()


class FaceRecognitionModule:
    """人脸识别模块类"""

    @staticmethod
    def _normalize_det_size(det_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        规范化 InsightFace 的 det_size。
        InsightFace/RetinaFace 检测器对输入尺寸有隐含要求：通常需要是 32 的倍数，否则可能出现
        内部 anchor/priors 尺寸不匹配，报类似 “operands could not be broadcast ...” 的错误。
        """
        try:
            w, h = det_size
        except Exception:
            return (640, 640)

        def _norm(x: int) -> int:
            try:
                xi = int(x)
            except Exception:
                xi = 640
            # 保底尺寸
            if xi < 160:
                xi = 160
            # 调整到最接近的 32 的倍数
            xi = int(round(xi / 32.0) * 32)
            if xi < 160:
                xi = 160
            return xi

        nw, nh = _norm(w), _norm(h)
        if (nw, nh) != (int(w), int(h)):
            print(f"⚠️  提示: det_size={det_size} 不是32的倍数，已自动调整为 ({nw}, {nh})（避免 InsightFace 检测器报错）")
        return (nw, nh)

    @staticmethod
    def _select_onnx_providers() -> List[str]:
        """
        选择 onnxruntime providers，优先使用 CUDA（如可用），否则回退 CPU。
        你也可以通过环境变量强制控制：
        - INSIGHTFACE_FORCE_CPU=1  -> 只用 CPU
        - INSIGHTFACE_USE_GPU=1    -> 优先 GPU（默认也是优先 GPU）
        - INSIGHTFACE_USE_TENSORRT=1 -> 允许尝试 TensorRT（默认不启用，避免缺少 TensorRT DLL 时刷屏报错）
        """
        force_cpu = os.environ.get("INSIGHTFACE_FORCE_CPU", "").strip() in {"1", "true", "True", "YES", "yes"}
        if force_cpu:
            return ["CPUExecutionProvider"]

        # 默认优先 GPU
        if ort is not None:
            try:
                avail = set(ort.get_available_providers())
                use_trt = os.environ.get("INSIGHTFACE_USE_TENSORRT", "").strip() in {"1", "true", "True", "YES", "yes"}
                # 默认优先顺序：CUDA -> DirectML -> CPU（避免无 TensorRT 环境时反复 LoadLibrary error 126）
                providers: List[str] = []
                if "CUDAExecutionProvider" in avail:
                    # Windows 上常见：onnxruntime-gpu 已安装，但缺少 CUDA/cuDNN/VC++ 运行时依赖，
                    # 导致加载 onnxruntime_providers_cuda.dll 报错 126 并刷屏。
                    # 这里做一次“可加载性”预检查：不能加载则不把 CUDAExecutionProvider 放进 providers。
                    if sys.platform == "win32":
                        try:
                            ort_mod = __import__("onnxruntime")
                            ort_pkg_dir = os.path.dirname(getattr(ort_mod, "__file__", "") or "")
                            dll_path = os.path.join(ort_pkg_dir, "capi", "onnxruntime_providers_cuda.dll")
                            if os.path.exists(dll_path):
                                ctypes.WinDLL(dll_path)  # 触发依赖解析
                                providers.append("CUDAExecutionProvider")
                            else:
                                # 找不到 dll，说明装的不是 gpu 包或安装不完整
                                pass
                        except Exception:
                            # 不能加载则跳过 CUDA，避免后续创建 Session 时刷屏报错
                            pass
                    else:
                        providers.append("CUDAExecutionProvider")
                # 仅当你显式开启时才尝试 TensorRT
                if use_trt and ("TensorrtExecutionProvider" in avail):
                    providers.insert(0, "TensorrtExecutionProvider")
                if "DmlExecutionProvider" in avail:
                    providers.append("DmlExecutionProvider")
                providers.append("CPUExecutionProvider")
                # 去重保持顺序
                seen = set()
                providers = [p for p in providers if not (p in seen or seen.add(p))]
                return providers
            except Exception:
                pass
        return ["CPUExecutionProvider"]
    
    def __init__(
        self,
        tolerance: float = 0.6,
        model_type: str = "insightface",
        det_size: Tuple[int, int] = (640, 640),
        det_size_small: Tuple[int, int] = (320, 320),
    ):
        """
        初始化人脸识别模块
        
        Args:
            tolerance: 人脸匹配阈值，越小越严格
                      - InsightFace: 0-2，推荐 0.3-0.4（使用余弦距离）
            model_type: 固定使用 'insightface'
            det_size: InsightFace 检测输入尺寸（越大越容易检测到远处/小人脸，但速度更慢；建议 640/800/960）
            det_size_small: 针对“小图参照照片”的备用 det_size（默认 320x320）
        """
        self.tolerance = tolerance
        self.model_type = (model_type or "insightface").lower()
        self.det_size = self._normalize_det_size(det_size)
        self.det_size_small = self._normalize_det_size(det_size_small)
        
        # 延迟导入 InsightFace（并保留对任务2 ROI 弹窗的兼容性）
        _ensure_insightface()
        
        # 检查模型可用性 - 强制使用 InsightFace
        if self.model_type != "insightface":
            raise ValueError("本项目已移除 face_recognition/dlib，仅支持 model_type='insightface'")

        if not INSIGHTFACE_AVAILABLE:
            # 这里必须提前退出；否则后续调用 FaceAnalysis 会触发 NameError（FaceAnalysis 未定义）
            hint = ""
            msg = (_INSIGHTFACE_IMPORT_ERR or "").lower()
            if "no module named 'albumentations'" in msg:
                hint = (
                    "\n可能原因：InsightFace 间接依赖 albumentations，但当前环境未安装。\n"
                    "修复（不会安装 opencv-python-headless，避免任务2弹窗失效）：\n"
                    "  python -m pip install albumentations==1.3.1 qudida==0.0.4 --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn\n"
                )
            elif "no module named 'qudida'" in msg:
                hint = (
                    "\n可能原因：缺少 qudida（albumentations 的依赖）。\n"
                    "修复（不会安装 opencv-python-headless，避免任务2弹窗失效）：\n"
                    "  python -m pip install qudida==0.0.4 --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn\n"
                )
            raise ImportError(
                "InsightFace 未安装或导入失败（FaceAnalysis 不可用）。\n"
                f"详细错误: {_INSIGHTFACE_IMPORT_ERR}\n"
                "基础安装：python -m pip install insightface onnxruntime\n"
                "GPU 可选：python -m pip install onnxruntime-gpu\n"
                + hint
                + "详细安装说明请查看: README.md"
            )
        
        # 初始化 InsightFace（如果使用）
        if self.model_type == "insightface":
            print(f"使用 InsightFace 模型 (tolerance={tolerance})")
            # 自动下载模型（如果不存在）
            # 优先尝试使用GPU，如果GPU不可用则自动回退到CPU
            providers = self._select_onnx_providers()
            self._providers = list(providers)
            if providers and providers[0] in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}:
                print(f"✅ InsightFace 将优先使用 GPU（{providers[0]}）")
            elif providers and providers[0] == "DmlExecutionProvider":
                print("✅ InsightFace 将优先使用 GPU（DmlExecutionProvider/DirectML）")
            else:
                print("ℹ️  InsightFace 将使用 CPU（CPUExecutionProvider）")

            # Windows 上常见问题：CUDA provider DLL 依赖缺失会触发 LoadLibrary error 126。
            # 这里做一次兜底：GPU 初始化失败则自动回退到 CPU，避免任务直接崩溃。
            try:
                # 重要：任务1只需要 detection+recognition。
                # InsightFace 的 landmark/genderage 在某些环境/输入上可能触发内部 NoneType shape 崩溃；
                # 禁用这些模块可显著提升稳定性（且更快），不影响出勤识别正确性。
                try:
                    self.face_app = FaceAnalysis(
                        name="buffalo_l",
                        providers=providers,
                        allowed_modules=["detection", "recognition"],
                    )
                except TypeError:
                    # 兼容旧版本 InsightFace（不支持 allowed_modules）
                    self.face_app = FaceAnalysis(
                        name="buffalo_l",
                        providers=providers,
                    )
                self.face_app.prepare(ctx_id=0, det_size=self.det_size)
            except Exception as e:
                # CUDA 常见 error 126：缺少 cudnn/cublas/VC++ 依赖
                print("⚠️  警告: InsightFace 尝试启用 GPU 失败，将自动回退。")
                print(f"    失败原因: {e}")
                # 如果 DirectML 可用，优先回退到 DML（仍然使用 GPU）
                fallback = ["DmlExecutionProvider", "CPUExecutionProvider"] if ("DmlExecutionProvider" in (providers or [])) else ["CPUExecutionProvider"]
                if fallback[0] == "DmlExecutionProvider":
                    print("ℹ️  回退到 DirectML（DmlExecutionProvider）继续使用 GPU")
                else:
                    print("ℹ️  回退到 CPU（CPUExecutionProvider）")
                try:
                    self.face_app = FaceAnalysis(name="buffalo_l", providers=fallback, allowed_modules=["detection", "recognition"])
                except TypeError:
                    self.face_app = FaceAnalysis(name="buffalo_l", providers=fallback)
                self.face_app.prepare(ctx_id=0, det_size=self.det_size)
            # 用于“小图参照照片”的备用检测器：
            # 经验上，det_size 过大且超过图片尺寸时，某些 512x512 证件照会出现检测为 0 的情况；
            # 使用更小的 det_size（如 320x320）可以显著提升参照图的人脸检测成功率。
            self._face_app_small = None
            self._face_app_cpu = None
            self._face_app_small_cpu = None
        # 不再支持 face_recognition 分支
        
        self.known_encodings = []
        self.known_student_ids = []
        self.known_names = []

    def _switch_to_cpu_face_apps(self):
        """
        DirectML 在某些 InsightFace 模型/节点上可能运行失败（例如 Reshape 报错 80070057）。
        发生时自动切换到 CPUExecutionProvider，保证流程可继续。
        """
        if self.model_type != "insightface":
            return
        if self._face_app_cpu is None:
            try:
                self._face_app_cpu = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
            except TypeError:
                self._face_app_cpu = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self._face_app_cpu.prepare(ctx_id=0, det_size=self.det_size)
        self.face_app = self._face_app_cpu
        # small app
        if self._face_app_small_cpu is None:
            try:
                self._face_app_small_cpu = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
            except TypeError:
                self._face_app_small_cpu = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self._face_app_small_cpu.prepare(ctx_id=0, det_size=self.det_size_small)
        # 注意：_face_app_small 只在需要时创建/使用；这里不强制覆盖，按需在调用处使用 cpu 版本即可。

    @staticmethod
    def _safe_face_get(face_app, img):
        """
        InsightFace FaceAnalysis.get 在某些版本/输入上可能在 landmark 分支触发
        AttributeError: 'NoneType' object has no attribute 'shape'。
        任务1只需要 detection+recognition，因此这里做兜底：遇到该错误直接当作“未检测到人脸”，
        避免单张坏样本/偶发 bug 导致整个流程崩溃。
        """
        try:
            return face_app.get(img)
        except AttributeError as e:
            msg = str(e)
            if "has no attribute 'shape'" in msg:
                return []
            raise
    
    def load_student_photos(self, photos_dir: str, student_list):
        """
        加载学生照片并生成人脸编码
        
        Args:
            photos_dir: 学生照片目录路径
            student_list: 学生信息字典，格式为 {student_id: {'name': name}}
                          或旧格式 {student_id: name}（向后兼容）
        """
        print(f"正在加载学生照片从: {photos_dir}")
        print("这可能需要一些时间，请稍候...")
        
        # 预先构建文件索引，避免重复遍历
        print("正在构建文件索引...")
        file_index = {}  # {文件名(不含扩展名): 完整路径}
        for root, dirs, files in os.walk(photos_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    name_without_ext = os.path.splitext(file)[0]
                    full_path = os.path.join(root, file)
                    # 如果同一个文件名有多个，保留第一个
                    if name_without_ext not in file_index:
                        file_index[name_without_ext] = full_path
        
        print(f"找到 {len(file_index)} 个照片文件")
        print("开始处理照片...")
        
        total = len(student_list)
        loaded_count = 0
        failed_count = 0
        
        for idx, (student_id, student_info) in enumerate(student_list.items(), 1):
            photo_path = None
            
            # 处理学生信息格式（支持新旧两种格式）
            if isinstance(student_info, dict):
                name = student_info.get('name', '')
            else:
                # 旧格式：直接是姓名
                name = student_info
            
            # 处理学号格式：如果是浮点数格式（如 23262010017.0），尝试去掉 .0
            student_id_clean = str(student_id).strip()
            if student_id_clean.endswith('.0'):
                student_id_clean = student_id_clean[:-2]  # 去掉 .0
            
            # 方法1: 尝试按学号查找（优先）- 直接检查文件索引
            if student_id_clean in file_index:
                photo_path = file_index[student_id_clean]
            elif str(student_id) in file_index:
                photo_path = file_index[str(student_id)]
            # 方法2: 尝试按姓名查找
            elif name in file_index:
                photo_path = file_index[name]
            # 方法3: 尝试部分匹配（学号）
            else:
                for key in file_index:
                    if key == student_id_clean or key == str(student_id) or \
                       key.startswith(f"{student_id_clean}.") or key.startswith(f"{student_id_clean}_"):
                        photo_path = file_index[key]
                        break
            # 方法4: 尝试部分匹配（姓名）
            if photo_path is None:
                for key in file_index:
                    if key == name or key.startswith(name):
                        photo_path = file_index[key]
                        break
            
            if photo_path is None or not os.path.exists(photo_path):
                failed_count += 1
                if failed_count <= 5:  # 只显示前5个失败的
                    print(f"  [{idx}/{total}] 警告: 未找到 {name} ({student_id}) 的照片")
                continue
            
            try:
                # 根据模型类型加载和处理图片
                if self.model_type == "insightface":
                    # 使用 InsightFace
                    img = cv2.imread(photo_path)
                    if img is None:
                        raise ValueError(f"无法读取图片: {photo_path}")
                    
                    # 首次检测
                    try:
                        faces = self._safe_face_get(self.face_app, img)
                    except Exception as e:
                        # DirectML 兼容性问题：切到 CPU 再试一次
                        if hasattr(self, "_providers") and ("DmlExecutionProvider" in (self._providers or [])):
                            print("⚠️  DirectML 推理失败，自动回退到 CPU 重新尝试（不影响结果，只是更慢）")
                            print(f"    原因: {e}")
                            self._switch_to_cpu_face_apps()
                            faces = self.face_app.get(img)
                        else:
                            raise

                    # 针对较小参照图（如 512x512）做一次“小 det_size”重试
                    if len(faces) == 0:
                        h, w = img.shape[:2]
                        if min(h, w) > 0 and min(h, w) <= 640:
                            if self._face_app_small is None:
                                providers = self._select_onnx_providers()
                                try:
                                    self._face_app_small = FaceAnalysis(
                                        name="buffalo_l",
                                        providers=providers,
                                        allowed_modules=["detection", "recognition"],
                                    )
                                    self._face_app_small.prepare(ctx_id=0, det_size=self.det_size_small)
                                except Exception:
                                    self._face_app_small = FaceAnalysis(
                                        name="buffalo_l",
                                        providers=["CPUExecutionProvider"],
                                        allowed_modules=["detection", "recognition"],
                                    )
                                    self._face_app_small.prepare(ctx_id=0, det_size=self.det_size_small)
                            try:
                                faces = self._safe_face_get(self._face_app_small, img)
                            except Exception as e:
                                if hasattr(self, "_providers") and ("DmlExecutionProvider" in (self._providers or [])):
                                    # small app 也回退 CPU 再试
                                    if self._face_app_small_cpu is None:
                                        try:
                                            self._face_app_small_cpu = FaceAnalysis(
                                                name="buffalo_l",
                                                providers=["CPUExecutionProvider"],
                                                allowed_modules=["detection", "recognition"],
                                            )
                                        except TypeError:
                                            self._face_app_small_cpu = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                                        self._face_app_small_cpu.prepare(ctx_id=0, det_size=self.det_size_small)
                                    faces = self._safe_face_get(self._face_app_small_cpu, img)
                                else:
                                    raise

                    # 若未检测到人脸，尝试对小图进行 2x 放大再检测（提升小脸/低分辨率的召回）
                    if len(faces) == 0:
                        h, w = img.shape[:2]
                        if h > 0 and w > 0 and max(h, w) < 800:
                            up = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
                            faces = self._safe_face_get(self.face_app, up)
                    
                    if len(faces) > 0:
                        # 选取面积最大的那张人脸（避免多脸场景选错/选到小脸）
                        face = max(
                            faces,
                            key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                        )
                        embedding = face.normed_embedding  # 归一化的特征向量
                        
                        self.known_encodings.append(embedding)
                        self.known_student_ids.append(student_id)
                        self.known_names.append(name)
                        loaded_count += 1
                        # 每10个显示一次进度，或者显示前几个
                        if loaded_count <= 3 or loaded_count % 10 == 0 or idx == total:
                            print(f"  [{idx}/{total}] 已加载: {name} ({student_id})")
                    else:
                        failed_count += 1
                        if failed_count <= 5:
                            print(f"  [{idx}/{total}] 警告: {name} 的照片中未检测到人脸")
                else:
                    raise RuntimeError("本项目已移除 face_recognition/dlib，仅支持 InsightFace。")
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:
                    print(f"  [{idx}/{total}] 错误: 加载 {name} 的照片时出错: {e}")
        
        print(f"\n加载完成: 成功 {loaded_count} 个, 失败 {failed_count} 个")
        print(f"总共加载了 {len(self.known_encodings)} 个学生的人脸编码")
    
    def detect_faces_in_frame(self, frame: np.ndarray) -> Tuple[List, List]:
        """
        在视频帧中检测人脸并生成特征向量
        
        Args:
            frame: 视频帧（BGR格式）
            
        Returns:
            face_locations: 人脸位置列表
            face_encodings: 人脸编码列表
        """
        if self.model_type == "insightface":
            # 使用 InsightFace 检测
            try:
                faces = self._safe_face_get(self.face_app, frame)
            except Exception as e:
                if hasattr(self, "_providers") and ("DmlExecutionProvider" in (self._providers or [])):
                    print("⚠️  DirectML 推理失败，自动回退到 CPU（帧检测）")
                    print(f"    原因: {e}")
                    self._switch_to_cpu_face_apps()
                    faces = self.face_app.get(frame)
                else:
                    raise
            
            face_locations = []
            face_encodings = []
            
            for face in faces:
                # InsightFace 返回的是 bbox (x1, y1, x2, y2) 格式
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                # 转换为 (top, right, bottom, left) 格式以保持兼容
                face_locations.append((y1, x2, y2, x1))
                
                # 获取归一化的特征向量
                embedding = face.normed_embedding
                face_encodings.append(embedding)
        else:
            raise RuntimeError("本项目已移除 face_recognition/dlib，仅支持 InsightFace。")
        
        return face_locations, face_encodings
    
    def match_faces(self, face_encodings: List) -> List[Optional[Tuple[str, str, float, Optional[float]]]]:
        """
        匹配检测到的人脸与已知学生
        
        Args:
            face_encodings: 检测到的人脸编码列表
            
        Returns:
            匹配结果列表，每个元素为 (student_id, name, best_distance, second_best_distance) 或 None
        """
        matches = []
        
        for i, face_encoding in enumerate(face_encodings):
            if len(self.known_encodings) == 0:
                matches.append(None)
                continue

            # 注意：之前使用启发式“属性过滤”做候选筛选，容易误判导致正确学生根本不参与匹配，
            # 从而大幅降低召回率。这里改为：默认对所有学生匹配（55人规模很小，性能可接受）。
            candidate_indices = list(range(len(self.known_encodings)))
            candidate_encodings = self.known_encodings
            
            if self.model_type == "insightface":
                # InsightFace 使用余弦距离（1 - 余弦相似度）
                # 特征向量已经归一化（normed_embedding），可以直接计算点积
                # 确保 face_encoding 也是归一化的
                face_encoding_norm = face_encoding / (np.linalg.norm(face_encoding) + 1e-10)
                # 计算余弦相似度（点积，因为都是归一化向量）
                similarities = np.dot(candidate_encodings, face_encoding_norm)
                # 转换为距离（1 - 相似度），距离越小越相似
                # 余弦相似度范围 [-1, 1]，转换为距离 [0, 2]
                distances = 1 - similarities
            else:
                raise RuntimeError("本项目已移除 face_recognition/dlib，仅支持 InsightFace。")
            
            # 找到最小距离
            min_distance_local_idx = np.argmin(distances)
            min_distance = distances[min_distance_local_idx]
            min_distance_global_idx = candidate_indices[min_distance_local_idx]

            # 第二小距离（用于“第一名/第二名差距”判断，避免相似度都差不多时误识别）
            second_best_distance: Optional[float] = None
            if len(distances) >= 2:
                # np.partition 是 O(n)，不需要全排序
                second_best_distance = float(np.partition(distances, 1)[1])
            
            # 如果距离小于阈值，则匹配成功
            if min_distance <= self.tolerance:
                matches.append((
                    self.known_student_ids[min_distance_global_idx],
                    self.known_names[min_distance_global_idx],
                    float(min_distance),
                    second_best_distance
                ))
            else:
                matches.append(None)
        
        return matches
    
    def process_video_frame(self, frame: np.ndarray) -> Dict[str, List]:
        """
        处理视频帧，返回检测到的学生信息
        
        Args:
            frame: 视频帧
            
        Returns:
            包含检测结果的字典
        """
        face_locations, face_encodings = self.detect_faces_in_frame(frame)
        matches = self.match_faces(face_encodings)
        
        detected_students = []
        for i, match in enumerate(matches):
            if match is not None:
                student_id, name, distance, second_best_distance = match
                confidence = 1 - distance  # InsightFace 下等价于 cosine similarity

                # 之前这里硬编码了 min_confidence=0.5（等价于距离<=0.5），会严重降低召回率，
                # 并且会覆盖用户设置的 tolerance（当 tolerance > 0.5 时尤其明显）。
                # 改为：min_confidence 与 tolerance 保持一致（InsightFace: distance=1-similarity）
                if self.model_type == "insightface":
                    min_confidence = max(0.0, min(1.0, 1.0 - float(self.tolerance)))
                    # 同时加一个“第一名/第二名差距”约束，降低误识别
                    # 距离越小越好，因此 margin = second_best - best
                    # 注意：在教室远景/小脸场景里，第一名与第二名距离往往非常接近，
                    # 过大的 margin 会把本来正确的匹配也过滤掉，导致“匹配不到学生”。
                    # 这里默认放宽到 0.0（不靠 margin 拦截），需要更严格时再调大（如 0.02~0.05）。
                    min_margin = 0.0
                    margin_ok = True
                    if second_best_distance is not None:
                        margin_ok = (float(second_best_distance) - float(distance)) >= min_margin
                else:
                    # face_recognition 仍沿用旧的最低置信度阈值
                    min_confidence = 0.5
                    margin_ok = True

                if confidence >= min_confidence and margin_ok:
                    top, right, bottom, left = face_locations[i]
                    detected_students.append({
                        'student_id': student_id,
                        'name': name,
                        'confidence': confidence,
                        'distance': distance,  # 保留距离信息用于调试
                        'location': (top, right, bottom, left)
                    })
        
        return {
            'detected_students': detected_students,
            'face_locations': face_locations,
            'total_faces': len(face_locations)
        }

