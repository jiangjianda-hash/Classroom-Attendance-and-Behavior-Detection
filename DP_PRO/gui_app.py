from __future__ import annotations

import io
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app_paths import resource_path
from main import parse_timepoint_to_minutes


def _maybe_set_offline_env_defaults() -> None:
    # InsightFace 离线模型目录（如果 assets 存在）
    ins_home = resource_path(os.path.join("assets", "insightface"))
    ins_models = os.path.join(ins_home, "models")
    if os.path.isdir(ins_models):
        os.environ.setdefault("INSIGHTFACE_HOME", ins_home)


def _set_insightface_force_cpu(force_cpu: bool) -> None:
    """
    控制任务1(InsightFace/onnxruntime)是否强制 CPU。
    - force_cpu=True：设置 INSIGHTFACE_FORCE_CPU=1，避免 GPU 依赖缺失导致报错/刷屏
    - force_cpu=False：移除 INSIGHTFACE_FORCE_CPU（若环境满足，可自动用 GPU provider）
    """
    if force_cpu:
        os.environ["INSIGHTFACE_FORCE_CPU"] = "1"
        os.environ.pop("INSIGHTFACE_USE_GPU", None)
    else:
        os.environ.pop("INSIGHTFACE_FORCE_CPU", None)
        os.environ["INSIGHTFACE_USE_GPU"] = "1"


def _ort_available_providers() -> list[str]:
    """
    返回当前环境 onnxruntime 可用的 providers。
    在 EXE 内如果打包的是 CPU 版 onnxruntime，通常只有 CPUExecutionProvider，
    此时“任务1优先使用GPU”的勾选只是偏好，实际会自动回退到 CPU。
    """
    try:
        import onnxruntime as ort  # type: ignore

        return list(ort.get_available_providers() or [])
    except Exception:
        return []


def _task1_gpu_is_available() -> bool:
    providers = set(_ort_available_providers())
    return ("CUDAExecutionProvider" in providers) or ("DmlExecutionProvider" in providers)


class _QtLogStream(QtCore.QObject):
    text = QtCore.Signal(str)

    def write(self, s: str) -> int:  # type: ignore[override]
        if not s:
            return 0
        self.text.emit(str(s))
        return len(s)

    def flush(self) -> None:  # noqa: D401
        return


def _read_frame_at_timepoint(video_path: str, timepoint: str) -> np.ndarray:
    """
    读取视频在指定时间点处的一帧（BGR）。
    注意：这里只做视频解码，不涉及任何系统命令/路径拼接，符合安全规则。
    """
    minute = float(parse_timepoint_to_minutes(timepoint))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0
    frame_idx = int(max(0.0, minute * 60.0 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"无法读取视频帧（timepoint={timepoint}, frame={frame_idx}）")
    return frame


class RoiSelectDialog(QtWidgets.QDialog):
    """
    Qt ROI 框选对话框：显示一帧图像，鼠标拖拽得到 ROI（原图坐标系）。
    这样不依赖 OpenCV HighGUI，打包后更稳定。
    """

    def __init__(self, frame_bgr: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("框选 ROI（拖拽选择区域）")
        self.resize(980, 720)

        self._frame_bgr = frame_bgr
        h, w = frame_bgr.shape[:2]
        self._orig_w = int(w)
        self._orig_h = int(h)

        # UI
        layout = QtWidgets.QVBoxLayout(self)
        self.view = QtWidgets.QLabel()
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setStyleSheet("background: #111;")
        layout.addWidget(self.view, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_ok = QtWidgets.QPushButton("确定")
        self.btn_cancel = QtWidgets.QPushButton("取消")
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_ok)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        # image
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], int(rgb.strides[0]), QtGui.QImage.Format_RGB888)
        self._pixmap = QtGui.QPixmap.fromImage(qimg)
        self.view.setPixmap(self._pixmap.scaled(self.view.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        # selection
        self._rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self.view)
        self._origin = QtCore.QPoint()
        self._sel_rect_view = QtCore.QRect()

        self._roi: Optional[Tuple[int, int, int, int]] = None

        # install mouse event filter
        self.view.installEventFilter(self)

    def resizeEvent(self, e):  # type: ignore[override]
        super().resizeEvent(e)
        if self._pixmap is not None:
            self.view.setPixmap(self._pixmap.scaled(self.view.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def eventFilter(self, obj, event):  # type: ignore[override]
        if obj is self.view:
            if event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                self._origin = event.pos()
                self._sel_rect_view = QtCore.QRect(self._origin, QtCore.QSize())
                self._rubber.setGeometry(self._sel_rect_view)
                self._rubber.show()
                return True
            if event.type() == QtCore.QEvent.MouseMove and self._rubber.isVisible():
                self._sel_rect_view = QtCore.QRect(self._origin, event.pos()).normalized()
                self._rubber.setGeometry(self._sel_rect_view)
                return True
            if event.type() == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton and self._rubber.isVisible():
                self._sel_rect_view = self._rubber.geometry()
                self._rubber.hide()
                self._roi = self._map_view_rect_to_image_roi(self._sel_rect_view)
                return True
        return super().eventFilter(obj, event)

    def _map_view_rect_to_image_roi(self, r: QtCore.QRect) -> Optional[Tuple[int, int, int, int]]:
        # 计算 view 中 pixmap 实际显示区域（考虑 KeepAspectRatio）
        if self._pixmap is None:
            return None
        vw, vh = self.view.width(), self.view.height()
        iw, ih = self._orig_w, self._orig_h
        if vw <= 0 or vh <= 0 or iw <= 0 or ih <= 0:
            return None
        scale = min(vw / iw, vh / ih)
        disp_w = int(iw * scale)
        disp_h = int(ih * scale)
        off_x = int((vw - disp_w) / 2)
        off_y = int((vh - disp_h) / 2)

        # clamp to displayed image area
        x1 = max(off_x, min(off_x + disp_w, int(r.left())))
        y1 = max(off_y, min(off_y + disp_h, int(r.top())))
        x2 = max(off_x, min(off_x + disp_w, int(r.right())))
        y2 = max(off_y, min(off_y + disp_h, int(r.bottom())))
        if x2 <= x1 or y2 <= y1:
            return None
        # map back to original image coords
        ox1 = int((x1 - off_x) / max(1e-6, scale))
        oy1 = int((y1 - off_y) / max(1e-6, scale))
        ox2 = int((x2 - off_x) / max(1e-6, scale))
        oy2 = int((y2 - off_y) / max(1e-6, scale))
        ox1 = max(0, min(iw - 1, ox1))
        oy1 = max(0, min(ih - 1, oy1))
        ox2 = max(ox1 + 1, min(iw, ox2))
        oy2 = max(oy1 + 1, min(ih, oy2))
        return (ox1, oy1, int(ox2 - ox1), int(oy2 - oy1))

    def get_roi(self) -> Optional[Tuple[int, int, int, int]]:
        return self._roi


@dataclass
class Task1Params:
    video: str
    photos: str
    student_list_csv: str
    timepoint: str
    tolerance: float
    det_size: int
    sample_interval: int
    max_frames: int
    prefer_gpu: bool


@dataclass
class Task2Params:
    video: str
    mode: str  # person/class
    timepoint: str
    duration_min: float
    sample_seconds: float
    device: str  # cpu
    roi: str  # "x,y,w,h" or ""
    interactive_roi: bool
    seat_map_path: str
    student_id: str
    pose_model: str
    obj_model: str
    imgsz: int
    obj_roi_imgsz: int
    obj_roi_max_people: int
    obj_min_conf: float
    save_marked: bool


class Worker(QtCore.QThread):
    finished_ok = QtCore.Signal(str)
    finished_err = QtCore.Signal(str)

    def __init__(self, kind: str, t1: Optional[Task1Params] = None, t2: Optional[Task2Params] = None):
        super().__init__()
        self.kind = kind
        self.t1 = t1
        self.t2 = t2

    def run(self) -> None:  # noqa: D401
        try:
            _maybe_set_offline_env_defaults()
            if self.kind == "task1":
                assert self.t1 is not None
                # 任务1 GPU 为可选：需要 onnxruntime-gpu + CUDA/cuDNN/VC++ DLL 依赖齐全
                _set_insightface_force_cpu(force_cpu=not bool(self.t1.prefer_gpu))
                # 在 GUI 日志里明确打印“EXE 实际可用的 providers”，避免看起来像按钮没生效
                providers = _ort_available_providers()
                print(f"[Task1] onnxruntime providers: {providers}")
                if self.t1.prefer_gpu and (not _task1_gpu_is_available()):
                    print("[Task1][WARN] 当前环境没有 CUDA/Dml provider，任务1将继续使用 CPU。")
                    print("[Task1][WARN] 如需 GPU：用 onnxruntime-gpu（NVIDIA/CUDA）或 onnxruntime-directml（DirectML）重新打包 EXE。")
                from main import load_student_list, task1_attendance_check

                student_list = load_student_list(self.t1.student_list_csv)
                start_min = float(parse_timepoint_to_minutes(self.t1.timepoint))
                report_df, absent = task1_attendance_check(
                    self.t1.video,
                    self.t1.photos,
                    student_list,
                    start_time_minutes=float(start_min),
                    tolerance=float(self.t1.tolerance),
                    det_size=int(self.t1.det_size),
                    sample_interval=int(self.t1.sample_interval),
                    max_frames=int(self.t1.max_frames),
                )
                total = len(student_list)
                absent_n = len(absent)
                present_n = max(0, total - absent_n)
                rate = (present_n / max(1, total)) if total else 0.0
                msg = f"任务1完成：出勤 {present_n}/{total}（出勤率 {rate:.2%}），缺勤 {absent_n} 人。输出：attendance_images/、attendance_report.csv"
                self.finished_ok.emit(msg)
                return

            if self.kind == "task2":
                assert self.t2 is not None
                from task2_behavior_flow import Task2Config, run_task2 as run_task2_flow

                roi_tuple: Optional[Tuple[int, int, int, int]] = None
                if self.t2.roi.strip():
                    from task2_behavior_flow import _parse_roi

                    roi_tuple = _parse_roi(self.t2.roi)

                cfg = Task2Config(
                    video_path=self.t2.video,
                    mode=str(self.t2.mode),
                    start_minute=float(parse_timepoint_to_minutes(self.t2.timepoint)),
                    duration_minutes=float(self.t2.duration_min),
                    sample_seconds=float(self.t2.sample_seconds),
                    tracker="CSRT",
                    roi=roi_tuple,
                    interactive_roi=bool(self.t2.interactive_roi),
                    output_dir="task2_outputs",
                    output_json="task2_results.json",
                    save_images=bool(self.t2.save_marked),
                    pose_model=str(self.t2.pose_model),
                    obj_model=str(self.t2.obj_model),
                    device=str(self.t2.device),
                    imgsz=int(self.t2.imgsz),
                    obj_every=1,
                    no_obj=False,
                    obj_roi_imgsz=int(self.t2.obj_roi_imgsz),
                    obj_roi_max_people=int(self.t2.obj_roi_max_people),
                    debug_kps=False,
                    min_iou_keep=0.12,
                    max_shift_px=90.0,
                    relocal_ema_alpha=0.25,
                    obj_min_iou=0.05,
                    obj_min_conf=float(self.t2.obj_min_conf),
                    roi_max_w=1920,
                    roi_max_h=1080,
                    seat_map_path=str(self.t2.seat_map_path),
                    student_id=str(self.t2.student_id).strip() or None,
                    images_max=200,
                    images_every=1,
                    pose_conf=0.25,
                    pose_iou=0.70,
                    pose_max_det=120,
                    pose_tiles="1,1",
                    pose_tile_dedupe_iou=0.55,
                    class_roi_augment=False,
                    class_roi_imgsz=0,
                    pose_min_upper_avg=0.10,
                    pose_min_upper_cnt=2,
                    pose_min_torso_cnt=1,
                    pose_dedupe_center_thr=0.35,
                    seat_assign_min_iou=0.05,
                    head_ema_alpha=0.35,
                )
                out = run_task2_flow(cfg)
                if out.get("error"):
                    raise RuntimeError(str(out))
                msg = f"任务2完成：结果 {out.get('output_json')}，汇总 {out.get('summary_json')}"
                self.finished_ok.emit(msg)
                return

            raise ValueError(f"未知任务类型: {self.kind}")
        except Exception as e:
            self.finished_err.emit(f"{e}\n\n{traceback.format_exc()}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DP_PRO 课堂分析（任务1/2）")
        self.resize(1100, 720)

        _maybe_set_offline_env_defaults()

        self._log_stream = _QtLogStream()
        self._log_stream.text.connect(self._append_log)

        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        # 注意：打包为 --noconsole 时 stdout 可能不具备 buffer，避免在这里包一层。

        self._worker: Optional[Worker] = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.tab1 = QtWidgets.QWidget()
        self.tab2 = QtWidgets.QWidget()
        self.tabs.addTab(self.tab1, "任务1：出勤检测")
        self.tabs.addTab(self.tab2, "任务2：课堂行为")

        self._build_task1_ui()
        self._build_task2_ui()

        # log panel
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        self.status = QtWidgets.QLabel("就绪（CPU-only，离线权重优先）")
        layout.addWidget(self.status)

    def closeEvent(self, event):  # type: ignore[override]
        try:
            sys.stdout = self._old_stdout
            sys.stderr = self._old_stderr
        except Exception:
            pass
        super().closeEvent(event)

    def _append_log(self, s: str) -> None:
        # 这里用 insertPlainText，保证“逐字符/逐块输出”也能展示（print 会分多次 write）
        cur = self.log.textCursor()
        cur.movePosition(QtGui.QTextCursor.End)
        cur.insertText(str(s))
        self.log.setTextCursor(cur)
        self.log.ensureCursorVisible()

    def _browse_file(self, line: QtWidgets.QLineEdit, filter_text: str) -> None:
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择文件", "", filter_text)
        if p:
            line.setText(p)

    def _browse_dir(self, line: QtWidgets.QLineEdit) -> None:
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "选择文件夹", "")
        if p:
            line.setText(p)

    def _build_task1_ui(self) -> None:
        form = QtWidgets.QFormLayout(self.tab1)

        self.t1_video = QtWidgets.QLineEdit()
        btn_v = QtWidgets.QPushButton("选择...")
        btn_v.clicked.connect(lambda: self._browse_file(self.t1_video, "视频文件 (*.mp4 *.avi *.mov *.mkv)"))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.t1_video, 1)
        row.addWidget(btn_v)
        form.addRow("视频", row)

        self.t1_photos = QtWidgets.QLineEdit()
        btn_p = QtWidgets.QPushButton("选择...")
        btn_p.clicked.connect(lambda: self._browse_dir(self.t1_photos))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.t1_photos, 1)
        row.addWidget(btn_p)
        form.addRow("学生照片目录", row)

        self.t1_list = QtWidgets.QLineEdit()
        btn_l = QtWidgets.QPushButton("选择...")
        btn_l.clicked.connect(lambda: self._browse_file(self.t1_list, "CSV 文件 (*.csv)"))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.t1_list, 1)
        row.addWidget(btn_l)
        form.addRow("student_list.csv", row)

        self.t1_timepoint = QtWidgets.QLineEdit("63")
        self.t1_timepoint.setPlaceholderText("例如：68 或 01:08:00 或 08:30")
        form.addRow("时间点", self.t1_timepoint)

        self.t1_tol = QtWidgets.QDoubleSpinBox()
        self.t1_tol.setRange(0.0, 2.0)
        self.t1_tol.setDecimals(3)
        self.t1_tol.setValue(0.90)
        form.addRow("tolerance", self.t1_tol)

        self.t1_det = QtWidgets.QSpinBox()
        self.t1_det.setRange(160, 2048)
        self.t1_det.setValue(640)
        form.addRow("det_size", self.t1_det)

        self.t1_interval = QtWidgets.QSpinBox()
        self.t1_interval.setRange(1, 1000000)
        self.t1_interval.setValue(500)
        form.addRow("sample_interval(帧)", self.t1_interval)

        self.t1_max = QtWidgets.QSpinBox()
        self.t1_max.setRange(1, 100000)
        self.t1_max.setValue(40)
        form.addRow("max_frames", self.t1_max)

        self.t1_prefer_gpu = QtWidgets.QCheckBox("任务1优先使用GPU（可选；需要 onnxruntime-gpu + CUDA/cuDNN/VC++ 运行库）")
        self.t1_prefer_gpu.setChecked(False)
        form.addRow(self.t1_prefer_gpu)
        # 如果当前环境（尤其是打包后的 EXE）没有 GPU provider，则直接禁用该选项，避免“看起来没用”
        providers = _ort_available_providers()
        has_gpu = ("CUDAExecutionProvider" in providers) or ("DmlExecutionProvider" in providers)
        if not has_gpu:
            self.t1_prefer_gpu.setChecked(False)
            self.t1_prefer_gpu.setEnabled(False)
            self.t1_prefer_gpu.setText("任务1优先使用GPU（当前不可用：仅 CPUExecutionProvider）")
            self.t1_prefer_gpu.setToolTip("当前 onnxruntime 没有 CUDA/Dml provider。CPU 版 EXE 属正常现象。")
        else:
            self.t1_prefer_gpu.setToolTip(f"当前可用 providers: {providers}")

        self.btn_run_t1 = QtWidgets.QPushButton("开始任务1")
        self.btn_run_t1.clicked.connect(self._run_task1)
        form.addRow(self.btn_run_t1)

    def _build_task2_ui(self) -> None:
        form = QtWidgets.QFormLayout(self.tab2)

        self.t2_video = QtWidgets.QLineEdit()
        btn_v = QtWidgets.QPushButton("选择...")
        btn_v.clicked.connect(lambda: self._browse_file(self.t2_video, "视频文件 (*.mp4 *.avi *.mov *.mkv)"))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.t2_video, 1)
        row.addWidget(btn_v)
        form.addRow("视频", row)

        self.t2_mode = QtWidgets.QComboBox()
        self.t2_mode.addItems(["person", "class"])
        form.addRow("模式", self.t2_mode)

        self.t2_timepoint = QtWidgets.QLineEdit("0")
        self.t2_timepoint.setPlaceholderText("例如：75 或 01:15:00 或 05:30")
        form.addRow("时间点", self.t2_timepoint)

        self.t2_dur = QtWidgets.QDoubleSpinBox()
        self.t2_dur.setRange(0.1, 9999.0)
        self.t2_dur.setValue(10.0)
        form.addRow("持续分钟（默认10）", self.t2_dur)

        self.t2_sample = QtWidgets.QDoubleSpinBox()
        self.t2_sample.setRange(0.5, 120.0)
        self.t2_sample.setValue(10.0)
        form.addRow("采样间隔(秒)", self.t2_sample)

        self.t2_device = QtWidgets.QComboBox()
        self.t2_device.addItems(["cpu", "cuda"])
        self.t2_device.setCurrentText("cpu")
        form.addRow("设备（任务2）", self.t2_device)

        self.t2_roi = QtWidgets.QLineEdit()
        self.t2_roi.setPlaceholderText('person模式可填：x,y,w,h（不填可用 seat_map+student_id）')
        form.addRow("ROI(可选)", self.t2_roi)

        self.t2_interactive_roi = QtWidgets.QCheckBox("在 GUI 中框选 ROI（推荐：不依赖 OpenCV 弹窗，仅 person）")
        self.t2_interactive_roi.setChecked(False)
        form.addRow(self.t2_interactive_roi)

        self.t2_student_id = QtWidgets.QLineEdit()
        self.t2_student_id.setPlaceholderText("person模式可选：学号（配合 seat_map.json 初始化ROI）")
        form.addRow("student_id(可选)", self.t2_student_id)

        self.t2_seat_map = QtWidgets.QLineEdit("attendance_images/seat_map.json")
        btn_s = QtWidgets.QPushButton("选择...")
        btn_s.clicked.connect(lambda: self._browse_file(self.t2_seat_map, "JSON 文件 (*.json)"))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.t2_seat_map, 1)
        row.addWidget(btn_s)
        form.addRow("seat_map.json(可选)", row)

        # 离线权重默认
        pose_default = resource_path(os.path.join("assets", "weights", "yolov8s-pose.pt"))
        obj_default = resource_path(os.path.join("assets", "weights", "rtdetr-l.pt"))
        self.t2_pose_model = QtWidgets.QLineEdit(pose_default if os.path.exists(pose_default) else "yolov8s-pose.pt")
        self.t2_obj_model = QtWidgets.QLineEdit(obj_default if os.path.exists(obj_default) else "rtdetr-l.pt")
        form.addRow("pose_model", self.t2_pose_model)
        form.addRow("obj_model", self.t2_obj_model)

        self.t2_imgsz = QtWidgets.QSpinBox()
        self.t2_imgsz.setRange(320, 1920)
        self.t2_imgsz.setValue(640)
        form.addRow("imgsz", self.t2_imgsz)

        self.t2_obj_roi_imgsz = QtWidgets.QSpinBox()
        self.t2_obj_roi_imgsz.setRange(0, 2048)
        self.t2_obj_roi_imgsz.setValue(960)
        form.addRow("桌面ROI二次检测imgsz", self.t2_obj_roi_imgsz)

        self.t2_obj_roi_max_people = QtWidgets.QSpinBox()
        self.t2_obj_roi_max_people.setRange(0, 200)
        self.t2_obj_roi_max_people.setValue(15)
        form.addRow("class每帧二次检测人数上限", self.t2_obj_roi_max_people)

        self.t2_obj_min_conf = QtWidgets.QDoubleSpinBox()
        self.t2_obj_min_conf.setRange(0.01, 0.99)
        self.t2_obj_min_conf.setDecimals(2)
        self.t2_obj_min_conf.setValue(0.35)
        form.addRow("obj_min_conf", self.t2_obj_min_conf)

        self.t2_save_marked = QtWidgets.QCheckBox("保存标记图（task2_outputs/marked_images）")
        self.t2_save_marked.setChecked(True)
        form.addRow(self.t2_save_marked)

        self.btn_run_t2 = QtWidgets.QPushButton("开始任务2（CPU）")
        self.btn_run_t2.clicked.connect(self._run_task2)
        form.addRow(self.btn_run_t2)

    def _set_running(self, running: bool) -> None:
        self.btn_run_t1.setEnabled(not running)
        self.btn_run_t2.setEnabled(not running)
        self.status.setText("运行中..." if running else "就绪")

    def _run_task1(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        p = Task1Params(
            video=self.t1_video.text().strip(),
            photos=self.t1_photos.text().strip(),
            student_list_csv=self.t1_list.text().strip(),
            timepoint=self.t1_timepoint.text().strip(),
            tolerance=float(self.t1_tol.value()),
            det_size=int(self.t1_det.value()),
            sample_interval=int(self.t1_interval.value()),
            max_frames=int(self.t1_max.value()),
            prefer_gpu=bool(self.t1_prefer_gpu.isChecked()),
        )
        try:
            parse_timepoint_to_minutes(p.timepoint)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "参数错误", f"任务1 时间点格式不正确：{e}")
            return
        if not p.video or not os.path.exists(p.video):
            QtWidgets.QMessageBox.warning(self, "参数错误", "请选择有效的视频文件")
            return
        if not p.photos or not os.path.isdir(p.photos):
            QtWidgets.QMessageBox.warning(self, "参数错误", "请选择有效的学生照片目录")
            return
        if not p.student_list_csv or not os.path.exists(p.student_list_csv):
            QtWidgets.QMessageBox.warning(self, "参数错误", "请选择有效的 student_list.csv")
            return

        self.log.appendPlainText("\n=== 开始任务1 ===\n")
        self._start_worker(Worker("task1", t1=p))

    def _run_task2(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        mode = self.t2_mode.currentText().strip()
        want_gui_roi = bool(self.t2_interactive_roi.isChecked()) if mode == "person" else False
        roi_text = self.t2_roi.text().strip()
        if mode == "person" and want_gui_roi and (not roi_text):
            # 用 Qt 框选得到 ROI（避免依赖 OpenCV HighGUI）
            try:
                frame = _read_frame_at_timepoint(self.t2_video.text().strip(), self.t2_timepoint.text().strip())
                dlg = RoiSelectDialog(frame, parent=self)
                if dlg.exec() != QtWidgets.QDialog.Accepted:
                    return
                roi = dlg.get_roi()
                if not roi:
                    QtWidgets.QMessageBox.warning(self, "ROI", "未选择有效 ROI，请重试。")
                    return
                roi_text = f"{roi[0]},{roi[1]},{roi[2]},{roi[3]}"
                self.t2_roi.setText(roi_text)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "ROI", f"框选 ROI 失败：{e}")
                return

        p = Task2Params(
            video=self.t2_video.text().strip(),
            mode=mode,
            timepoint=self.t2_timepoint.text().strip(),
            duration_min=float(self.t2_dur.value()),
            sample_seconds=float(self.t2_sample.value()),
            device=str(self.t2_device.currentText().strip() or "cpu"),
            roi=roi_text,
            interactive_roi=False,
            seat_map_path=self.t2_seat_map.text().strip(),
            student_id=self.t2_student_id.text().strip(),
            pose_model=self.t2_pose_model.text().strip(),
            obj_model=self.t2_obj_model.text().strip(),
            imgsz=int(self.t2_imgsz.value()),
            obj_roi_imgsz=int(self.t2_obj_roi_imgsz.value()),
            obj_roi_max_people=int(self.t2_obj_roi_max_people.value()),
            obj_min_conf=float(self.t2_obj_min_conf.value()),
            save_marked=bool(self.t2_save_marked.isChecked()),
        )
        if not p.video or not os.path.exists(p.video):
            QtWidgets.QMessageBox.warning(self, "参数错误", "请选择有效的视频文件")
            return
        try:
            parse_timepoint_to_minutes(p.timepoint)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "参数错误", f"任务2 时间点格式不正确：{e}")
            return
        if p.mode == "person" and (not p.roi and not p.student_id):
            QtWidgets.QMessageBox.warning(self, "参数错误", "person 模式需要 ROI（x,y,w,h）或 student_id+seat_map.json")
            return
        if p.student_id and (not p.seat_map_path or not os.path.exists(p.seat_map_path)):
            QtWidgets.QMessageBox.warning(self, "参数错误", "提供 student_id 时需要有效的 seat_map.json")
            return

        self.log.appendPlainText("\n=== 开始任务2 ===\n")
        self._start_worker(Worker("task2", t2=p))

    def _start_worker(self, w: Worker) -> None:
        self._worker = w
        self._set_running(True)

        # 把 worker 期间的 stdout/stderr 重定向到 GUI
        sys.stdout = self._log_stream  # type: ignore
        sys.stderr = self._log_stream  # type: ignore

        w.finished_ok.connect(self._on_ok)
        w.finished_err.connect(self._on_err)
        w.finished.connect(lambda: self._on_done())
        w.start()

    def _on_done(self) -> None:
        # 恢复
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr
        self._set_running(False)

    def _on_ok(self, msg: str) -> None:
        self.log.appendPlainText("\n✅ " + msg + "\n")
        QtWidgets.QMessageBox.information(self, "完成", msg)

    def _on_err(self, msg: str) -> None:
        self.log.appendPlainText("\n❌ 失败：\n" + msg + "\n")
        QtWidgets.QMessageBox.critical(self, "失败", msg)


def main() -> None:
    _maybe_set_offline_env_defaults()
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


