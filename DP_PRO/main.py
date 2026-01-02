"""
课堂行为分析系统主程序（仅任务1/任务2）
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from typing import Dict

import cv2
import pandas as pd

from task2_behavior_flow import Task2Config as Task2FlowConfig, run_task2 as run_task2_flow

# PyInstaller 打包提示：
# 任务1依赖在函数内动态 import（便于给出更友好的缺包提示），但 PyInstaller 的静态分析可能漏掉本地模块，
# 导致 EXE 运行时报 “No module named 'face_recognition_module'”。
# 这里用一次“安全的可选导入”让 PyInstaller 确认这些模块需要被打包。
try:  # pragma: no cover
    import face_recognition_module as _fr_mod  # noqa: F401
    import attendance_checker as _ac_mod  # noqa: F401
except Exception:
    # 开发环境/精简环境下允许忽略；真正运行任务1时会在 task1_attendance_check 里给出明确提示
    pass


# 设置标准输出编码为 UTF-8（处理中文路径）
if sys.platform == "win32":
    # 注意：PyInstaller 的 windowed 模式（console=False）下，sys.stdout/sys.stderr 可能为 None。
    # 因此这里必须做防御性判断，避免导入 main.py 就直接崩溃。
    try:
        if sys.stdout is not None and getattr(sys.stdout, "buffer", None) is not None:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass
    try:
        if sys.stderr is not None and getattr(sys.stderr, "buffer", None) is not None:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass


def _maybe_use_offline_assets() -> None:
    """
    若项目目录下存在 assets/，则优先启用离线模型目录：
    - InsightFace：设置 INSIGHTFACE_HOME=assets/insightface（其中应包含 models/）
    任务2权重路径则在解析参数后做一次“默认名 -> assets/weights/xxx.pt”的替换。
    """
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        ins_home = os.path.join(root, "assets", "insightface")
        ins_models = os.path.join(ins_home, "models")
        if os.path.isdir(ins_models):
            os.environ.setdefault("INSIGHTFACE_HOME", ins_home)
    except Exception:
        pass


def get_student_name(student_list: Dict, student_id: str) -> str:
    """从 student_list 中获取姓名（兼容 dict/str 两种形式）。"""
    if not student_id or student_id not in student_list:
        return "未知"
    info = student_list[student_id]
    if isinstance(info, dict):
        return str(info.get("name", "未知"))
    return str(info)


def load_student_list(csv_path: str) -> Dict[str, Dict[str, str]]:
    """从 CSV 文件加载选课名单（至少两列：学号/姓名）。"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"选课名单文件不存在: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    id_col = None
    name_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if "学号" in str(col) or "student_id" in col_lower or col_lower == "id":
            id_col = col
        if "姓名" in str(col) or "name" in col_lower:
            name_col = col
    if id_col is None or name_col is None:
        id_col = df.columns[0]
        name_col = df.columns[1]

    out: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        sid = str(row[id_col]).strip()
        name = str(row[name_col]).strip()
        if sid and name:
            out[sid] = {"name": name}
    print(f"已加载 {len(out)} 个学生的信息")
    return out


def parse_timepoint_to_minutes(s: str) -> float:
    """
    将“时间点”解析为分钟数（float）。支持：
    - "68" / "68.5"（分钟）
    - "MM:SS"
    - "HH:MM:SS"

    说明：该函数只做数值解析，不做任何路径/命令拼接，符合安全规则。
    """
    if s is None:
        return 0.0
    t = str(s).strip()
    if not t:
        return 0.0

    # 纯数字：按“分钟”处理
    try:
        return float(t)
    except Exception:
        pass

    parts = [p.strip() for p in t.split(":")]
    if len(parts) not in (2, 3):
        raise ValueError(f"时间点格式不支持: {s}（支持 分钟 或 HH:MM:SS 或 MM:SS）")

    try:
        nums = [float(p) for p in parts]
    except Exception as e:
        raise ValueError(f"时间点包含非数字: {s}") from e

    if len(nums) == 2:
        mm, ss = nums
        hh = 0.0
    else:
        hh, mm, ss = nums

    if hh < 0 or mm < 0 or ss < 0:
        raise ValueError(f"时间点不能为负数: {s}")
    if ss >= 60 or mm >= 60:
        # 允许用户写 90:00 这种“分钟:秒”吗？这里严格一点，避免歧义。
        raise ValueError(f"时间点分/秒需 < 60: {s}")

    return float(hh * 60.0 + mm + ss / 60.0)


def task1_attendance_check(
    video_path: str,
    photos_dir: str,
    student_list: Dict[str, Dict[str, str]],
    start_time_minutes: float = 63.0,
    tolerance: float = 0.9,
    det_size: int = 640,
    sample_interval: int = 500,
    max_frames: int = 40,
    output_path: str = "attendance_report.csv",
):
    """任务1：出勤检测 + 生成 seat_map.json。"""
    print("\n" + "=" * 60)
    print("任务1: 出勤检测")
    print("=" * 60)

    try:
        from face_recognition_module import FaceRecognitionModule
        from attendance_checker import AttendanceChecker
    except Exception as e:
        raise ImportError(f"导入任务1依赖失败: {e}") from e

    try:
        face_module = FaceRecognitionModule(
            tolerance=float(tolerance),
            model_type="insightface",
            det_size=(int(det_size), int(det_size)),
        )
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ 错误: InsightFace 未安装或初始化失败")
        print("=" * 60)
        print(f"错误信息: {e}")
        print("\n请查看 README.md 的 InsightFace 安装说明")
        print("=" * 60)
        raise

    face_module.load_student_photos(photos_dir, student_list)
    checker = AttendanceChecker(face_module)

    out_dir = "attendance_images"
    os.makedirs(out_dir, exist_ok=True)

    result = checker.check_attendance_from_video(
        video_path,
        sample_interval=int(sample_interval),
        max_frames=int(max_frames),
        start_time_minutes=float(start_time_minutes),
        output_images_dir=out_dir,
    )
    report_df = checker.generate_attendance_report(student_list, output_path)

    all_ids = set(student_list.keys())
    absent = checker.get_absent_students(all_ids)
    print(f"\n缺勤学生名单 ({len(absent)} 人):")
    for sid in absent:
        print(f"  - {get_student_name(student_list, sid)} ({sid})")

    print(f"\n输出目录: {out_dir}/")
    print(f"报告: {output_path}")
    print(f"处理帧数: {result.get('total_frames_processed')}")
    return report_df, absent


def main():
    def _resolve_video_path(video_path: str) -> str:
        """仅在项目目录内按文件名尝试定位视频（避免使用任意输入拼接路径）。"""
        if not video_path:
            return video_path
        if os.path.exists(video_path):
            return video_path
        base = os.path.basename(video_path)
        ext = os.path.splitext(base)[1].lower()
        if ext not in {".mp4", ".avi", ".mov", ".mkv"}:
            return video_path
        root = os.path.abspath(os.getcwd())
        for dirpath, _, filenames in os.walk(root):
            if os.path.basename(dirpath) in {"__pycache__", "attendance_images", "task2_outputs"}:
                continue
            if base in filenames:
                cand = os.path.abspath(os.path.join(dirpath, base))
                if cand.startswith(root):
                    print(f"⚠️  提示: 视频路径不存在，已自动定位为: {cand}")
                    return cand
        return video_path

    parser = argparse.ArgumentParser(description="课堂行为分析系统（任务1/任务2）")
    parser.add_argument("--video", type=str, required=True, help="教室视频路径")
    parser.add_argument("--task", type=int, choices=[1, 2], required=True, help="选择任务 (1:出勤检测, 2:课堂行为)")

    # 任务1
    parser.add_argument("--photos", type=str, default="", help="学生参照照片目录（任务1必填）")
    parser.add_argument("--list", type=str, default="", help="选课名单CSV（任务1必填）")
    parser.add_argument("--tolerance", type=float, default=0.9, help="任务1: 匹配阈值（InsightFace）")
    parser.add_argument("--det-size", type=int, default=640, help="任务1: InsightFace det_size")
    parser.add_argument("--start-time", type=str, default="63.0", help="任务1: 时间点（分钟 或 HH:MM:SS 或 MM:SS）")
    parser.add_argument("--sample-interval", type=int, default=500, help="任务1: 采样间隔（帧）")
    parser.add_argument("--max-frames", type=int, default=40, help="任务1: 最大采样次数")

    # 任务2（重做版）
    parser.add_argument("--student-id", type=str, default="", help="任务2: 目标学生ID（可用于从 seat_map 初始化 ROI）")
    parser.add_argument("--t2-start-minute", type=str, default="0.0", help="任务2: 时间点（分钟 或 HH:MM:SS 或 MM:SS）")
    parser.add_argument("--t2-duration", type=float, default=10.0)
    parser.add_argument("--t2-sample-seconds", type=float, default=10.0)
    parser.add_argument("--t2-mode", type=str, choices=["person", "class"], default="person")
    parser.add_argument("--t2-roi", type=str, default="")
    parser.add_argument("--t2-interactive-roi", action="store_true")
    parser.add_argument("--t2-pose-model", type=str, default="yolov8s-pose.pt")
    parser.add_argument("--t2-obj-model", type=str, default="rtdetr-l.pt")
    parser.add_argument("--t2-device", type=str, default="auto")
    parser.add_argument("--t2-imgsz", type=int, default=640)
    parser.add_argument("--t2-obj-every", type=int, default=1)
    parser.add_argument("--t2-no-obj", action="store_true")
    parser.add_argument(
        "--t2-obj-roi-imgsz",
        type=int,
        default=960,
        help="任务2(person): 桌面ROI二次物体检测输入尺寸（越大越容易检出斜拍小电脑，但更慢；<=0禁用）",
    )
    parser.add_argument(
        "--t2-obj-roi-max-people",
        type=int,
        default=15,
        help="任务2(class): 每个采样帧最多对多少个“低头”的人做桌面ROI二次物体检测（防止太慢）",
    )
    parser.add_argument("--t2-debug-kps", action="store_true")
    parser.add_argument("--t2-min-iou-keep", type=float, default=0.12)
    parser.add_argument("--t2-max-shift-px", type=float, default=90.0)
    parser.add_argument("--t2-relocal-ema-alpha", type=float, default=0.25)
    parser.add_argument("--t2-obj-min-iou", type=float, default=0.05)
    parser.add_argument("--t2-obj-min-conf", type=float, default=0.35)
    parser.add_argument("--t2-roi-max-w", type=int, default=1920)
    parser.add_argument("--t2-roi-max-h", type=int, default=1080)
    parser.add_argument("--t2-output-dir", type=str, default="task2_outputs")
    parser.add_argument("--t2-output-json", type=str, default="task2_results.json")
    parser.add_argument("--t2-save-marked-images", action="store_true")
    parser.add_argument("--t2-marked-max", type=int, default=200)
    parser.add_argument("--t2-marked-every", type=int, default=1)
    parser.add_argument("--t2-seat-map-path", type=str, default="attendance_images/seat_map.json")
    parser.add_argument("--t2-tracker", type=str, default="CSRT", choices=["CSRT", "KCF"])

    parser.add_argument("--t2-pose-conf", type=float, default=0.25)
    parser.add_argument("--t2-pose-iou", type=float, default=0.70)
    parser.add_argument("--t2-pose-max-det", type=int, default=120)
    parser.add_argument("--t2-pose-tiles", type=str, default="1,1")
    parser.add_argument("--t2-pose-tile-dedupe-iou", type=float, default=0.55)
    parser.add_argument("--t2-class-roi-augment", action="store_true")
    parser.add_argument("--t2-class-roi-imgsz", type=int, default=0)
    parser.add_argument("--t2-pose-min-upper-avg", type=float, default=0.10)
    parser.add_argument("--t2-pose-min-upper-cnt", type=int, default=2)
    parser.add_argument("--t2-pose-min-torso-cnt", type=int, default=1)
    parser.add_argument("--t2-pose-dedupe-center-thr", type=float, default=0.35)
    parser.add_argument("--t2-seat-assign-min-iou", type=float, default=0.05)
    parser.add_argument("--t2-head-ema-alpha", type=float, default=0.35)

    args = parser.parse_args()
    args.video = _resolve_video_path(args.video)
    _maybe_use_offline_assets()

    if args.task == 1:
        if not args.photos or not args.list:
            parser.error("任务1需要 --photos 与 --list")
        student_list = load_student_list(args.list)
        start_min = parse_timepoint_to_minutes(args.start_time)
        task1_attendance_check(
            args.video,
            args.photos,
            student_list,
            start_time_minutes=float(start_min),
            tolerance=float(args.tolerance),
            det_size=int(args.det_size),
            sample_interval=int(args.sample_interval),
            max_frames=int(args.max_frames),
        )
        return

    # task2
    # 若用户未显式传模型路径，且 assets/weights 存在，则使用离线权重（避免联网下载）
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        wdir = os.path.join(root, "assets", "weights")
        if os.path.isdir(wdir):
            if str(args.t2_pose_model).strip() == "yolov8s-pose.pt":
                cand = os.path.join(wdir, "yolov8s-pose.pt")
                if os.path.exists(cand):
                    args.t2_pose_model = cand
            if str(args.t2_obj_model).strip() == "rtdetr-l.pt":
                cand = os.path.join(wdir, "rtdetr-l.pt")
                if os.path.exists(cand):
                    args.t2_obj_model = cand
    except Exception:
        pass

    roi_tuple = None
    if args.t2_roi:
        from task2_behavior_flow import _parse_roi as _parse_roi

        roi_tuple = _parse_roi(args.t2_roi)

    cfg = Task2FlowConfig(
        video_path=args.video,
        mode=str(args.t2_mode),
        start_minute=float(parse_timepoint_to_minutes(args.t2_start_minute)),
        duration_minutes=float(args.t2_duration),
        sample_seconds=float(args.t2_sample_seconds),
        tracker=str(args.t2_tracker),
        roi=roi_tuple,
        interactive_roi=bool(args.t2_interactive_roi),
        output_dir=str(args.t2_output_dir),
        output_json=str(args.t2_output_json),
        save_images=bool(args.t2_save_marked_images),
        pose_model=str(args.t2_pose_model),
        obj_model=str(args.t2_obj_model),
        device=str(args.t2_device),
        imgsz=int(args.t2_imgsz),
        obj_every=int(args.t2_obj_every),
        no_obj=bool(args.t2_no_obj),
        obj_roi_imgsz=int(args.t2_obj_roi_imgsz),
        obj_roi_max_people=int(args.t2_obj_roi_max_people),
        debug_kps=bool(args.t2_debug_kps),
        min_iou_keep=float(args.t2_min_iou_keep),
        max_shift_px=float(args.t2_max_shift_px),
        relocal_ema_alpha=float(args.t2_relocal_ema_alpha),
        obj_min_iou=float(args.t2_obj_min_iou),
        obj_min_conf=float(args.t2_obj_min_conf),
        roi_max_w=int(args.t2_roi_max_w),
        roi_max_h=int(args.t2_roi_max_h),
        seat_map_path=str(args.t2_seat_map_path),
        student_id=str(args.student_id).strip() or None,
        images_max=int(args.t2_marked_max),
        images_every=int(args.t2_marked_every),
        pose_conf=float(args.t2_pose_conf),
        pose_iou=float(args.t2_pose_iou),
        pose_max_det=int(args.t2_pose_max_det),
        pose_tiles=str(args.t2_pose_tiles),
        pose_tile_dedupe_iou=float(args.t2_pose_tile_dedupe_iou),
        class_roi_augment=bool(args.t2_class_roi_augment),
        class_roi_imgsz=int(args.t2_class_roi_imgsz),
        pose_min_upper_avg=float(args.t2_pose_min_upper_avg),
        pose_min_upper_cnt=int(args.t2_pose_min_upper_cnt),
        pose_min_torso_cnt=int(args.t2_pose_min_torso_cnt),
        pose_dedupe_center_thr=float(args.t2_pose_dedupe_center_thr),
        seat_assign_min_iou=float(args.t2_seat_assign_min_iou),
        head_ema_alpha=float(args.t2_head_ema_alpha),
    )

    if cfg.mode == "person" and (cfg.roi is None and not cfg.interactive_roi and not cfg.student_id):
        raise ValueError("任务2(person) 需要 ROI：使用 --t2-roi 或 --t2-interactive-roi，或提供 --student-id 且 seat_map 存在")

    out = run_task2_flow(cfg)
    if out.get("error") == "missing_ultralytics_or_torch":
        return
    print(f"✅ 任务2完成：samples={out['n']}")
    print(f"结果JSON: {out['output_json']}")
    if out.get("summary_json"):
        print(f"汇总JSON: {out['summary_json']}")
    if out.get("images_dir"):
        print(f"标记图目录: {out['images_dir']}")


if __name__ == "__main__":
    main()


