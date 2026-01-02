"""
任务2（重做版）行为分析流程（按用户流程图）

模式：
- person（个人）：框选目标 -> CSRT 跟踪 + Pose -> 规则引擎
- class（全班）：每10秒取1帧 -> YOLOv8n-pose + RT-DETR-L -> 规则引擎

依赖：
- ultralytics（用于 YOLOv8n-pose / RT-DETR-L），未安装则提示用户安装。
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from behavior_rules_yolo import DetBox, infer_behaviors, _iou


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_roi(roi_str: str) -> Tuple[int, int, int, int]:
    parts = [p.strip() for p in roi_str.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI 格式应为 x,y,w,h")
    x, y, w, h = [int(float(p)) for p in parts]
    return x, y, w, h


def _clamp_roi(roi: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x, y, rw, rh = [int(v) for v in roi]
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    rw = max(1, min(w - x, rw))
    rh = max(1, min(h - y, rh))
    return x, y, rw, rh


def _parse_tiles(s: str) -> Tuple[int, int]:
    """
    解析 "cols,rows"（例如 "2,2"），用于 class 模式 pose 分块推理。
    """
    if not s:
        return (1, 1)
    parts = [p.strip() for p in str(s).split(",")]
    if len(parts) != 2:
        return (1, 1)
    try:
        c = int(float(parts[0]))
        r = int(float(parts[1]))
        c = max(1, min(8, c))
        r = max(1, min(8, r))
        return (c, r)
    except Exception:
        return (1, 1)


def _nms_xyxy(boxes: List[Tuple[float, float, float, float]], scores: List[float], iou_th: float) -> List[int]:
    """
    简单贪心NMS：返回保留的索引（按score从高到低）。
    """
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: float(scores[i]), reverse=True)
    keep: List[int] = []
    for i in order:
        bi = boxes[i]
        ok = True
        for j in keep:
            if float(_iou(bi, boxes[j])) >= float(iou_th):
                ok = False
                break
        if ok:
            keep.append(i)
    return keep


_UPPER_KPS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder")
_TORSO_KPS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


def _upper_kps_score(kps: Dict[str, Tuple[float, float, float]]) -> Tuple[float, int, int]:
    """
    返回 (upper_avg_conf, upper_cnt, torso_cnt)。
    upper_cnt: 上半身关键点中 conf>0 的数量
    torso_cnt: 躯干关键点中 conf>0 的数量（坐姿下腿常被遮挡，所以不要求腿）
    """
    if not kps:
        return (0.0, 0, 0)
    upper_confs = [float(kps[n][2]) for n in _UPPER_KPS if n in kps and float(kps[n][2]) > 0.0]
    torso_cnt = sum(1 for n in _TORSO_KPS if n in kps and float(kps[n][2]) > 0.0)
    if not upper_confs:
        return (0.0, 0, torso_cnt)
    return (float(sum(upper_confs) / max(1, len(upper_confs))), int(len(upper_confs)), int(torso_cnt))


def _filter_pose_candidates(
    boxes: List[Tuple[float, float, float, float]],
    scores: List[float],
    kps_list: List[Dict[str, Tuple[float, float, float]]],
    min_upper_avg: float = 0.10,
    min_upper_cnt: int = 2,
    min_torso_cnt: int = 1,
) -> Tuple[List[Tuple[float, float, float, float]], List[float], List[Dict[str, Tuple[float, float, float]]]]:
    """
    过滤 pose 产生的“假人框”（常见：书包/杂物被误检）。
    只依赖上半身/躯干关键点，不要求腿（教室坐姿腿经常不可见）。
    """
    out_b: List[Tuple[float, float, float, float]] = []
    out_s: List[float] = []
    out_k: List[Dict[str, Tuple[float, float, float]]] = []
    for b, s, k in zip(boxes, scores, kps_list):
        upper_avg, upper_cnt, torso_cnt = _upper_kps_score(k)
        if upper_cnt < int(min_upper_cnt):
            continue
        if torso_cnt < int(min_torso_cnt):
            continue
        if float(upper_avg) < float(min_upper_avg):
            continue
        # 用上半身置信度微调 score，便于后续去重时优先保留“更像人”的框
        new_s = float(s) * (0.5 + 0.5 * float(upper_avg))
        out_b.append(b)
        out_s.append(new_s)
        out_k.append(k)
    return (out_b, out_s, out_k)


def _dedupe_by_center(
    boxes: List[Tuple[float, float, float, float]],
    scores: List[float],
    kps_list: List[Dict[str, Tuple[float, float, float]]],
    center_thr_ratio: float = 0.35,
) -> Tuple[List[Tuple[float, float, float, float]], List[float], List[Dict[str, Tuple[float, float, float]]]]:
    """
    处理“一个人多个框但 IoU 不高”的情况（例如一个框偏头部、一个框偏躯干）。
    规则（按分数从高到低保留）：
    - 若小框几乎完全落在大框里（常见：头框落在身体框里），直接视为重复
    - 否则使用中心距离去重：中心点距离 < center_thr_ratio * min_side(更小框) 视为重复
    注意：center_thr_ratio 越大，合并越激进（更少“一人多框”）。
    """
    if len(boxes) <= 1:
        return (boxes, scores, kps_list)

    def _area(b: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = b
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

    def _inter_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(float(ax1), float(bx1))
        iy1 = max(float(ay1), float(by1))
        ix2 = min(float(ax2), float(bx2))
        iy2 = min(float(ay2), float(by2))
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        return float(iw * ih)

    def _anchor_xy(kps: Dict[str, Tuple[float, float, float]], box: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """
        用上半身关键点作为“同一人”锚点（更适配：头框 vs 身体框）。
        优先 nose，其次左右肩中点，最后回退到框中心。
        """
        if kps:
            n = kps.get("nose")
            if n is not None and float(n[2]) > 0.0:
                return (float(n[0]), float(n[1]))
            ls = kps.get("left_shoulder")
            rs = kps.get("right_shoulder")
            if ls is not None and rs is not None and float(ls[2]) > 0.0 and float(rs[2]) > 0.0:
                return (0.5 * (float(ls[0]) + float(rs[0])), 0.5 * (float(ls[1]) + float(rs[1])))
        x1, y1, x2, y2 = box
        return (0.5 * (float(x1) + float(x2)), 0.5 * (float(y1) + float(y2)))

    order = sorted(range(len(boxes)), key=lambda i: float(scores[i]), reverse=True)
    keep: List[int] = []
    for i in order:
        x1, y1, x2, y2 = boxes[i]
        ax, ay = _anchor_xy(kps_list[i] if i < len(kps_list) else {}, boxes[i])
        ok = True
        for j in keep:
            a1, b1, a2, b2 = boxes[j]
            bx, by = _anchor_xy(kps_list[j] if j < len(kps_list) else {}, boxes[j])
            dx = float(ax - bx)
            dy = float(ay - by)
            # 1) 包含关系去重：小框几乎完全在大框内
            inter = _inter_area(boxes[i], boxes[j])
            ai = _area(boxes[i])
            aj = _area(boxes[j])
            if ai > 0.0 and aj > 0.0:
                small = min(ai, aj)
                if small > 0.0 and float(inter / small) >= 0.80:
                    ok = False
                    break

            # 2) 中心距离去重：阈值按“更小框”的尺度算，更适配“头框 vs 身体框”
            wi = max(1.0, float(x2 - x1))
            hi = max(1.0, float(y2 - y1))
            wj = max(1.0, float(a2 - a1))
            hj = max(1.0, float(b2 - b1))
            thr = float(center_thr_ratio) * float(min(min(wi, hi), min(wj, hj)))
            if (dx * dx + dy * dy) ** 0.5 <= thr:
                ok = False
                break
        if ok:
            keep.append(i)
    return ([boxes[i] for i in keep], [scores[i] for i in keep], [kps_list[i] for i in keep])


def _infer_pose_boxes_kps_for_region(
    backend: "UltralyticsBackend",
    frame_bgr: "np.ndarray",
    region_xywh: Tuple[int, int, int, int],
    tiles_c: int,
    tiles_r: int,
    pose_conf: float,
    pose_iou: float,
    pose_max_det: int,
    pose_imgsz: int,
) -> Tuple[List[Tuple[float, float, float, float]], List[float], List[Dict[str, Tuple[float, float, float]]]]:
    """
    对指定 region (x,y,w,h) 做 pose 推理。支持 tiles 分块以提升远处小人召回。
    返回：boxes(已映射回全图坐标), scores, kps(已映射回全图坐标)。
    """
    x, y, w, h = region_xywh
    crop = frame_bgr[y : y + h, x : x + w]
    if crop.size == 0:
        return ([], [], [])

    det_boxes: List[Tuple[float, float, float, float]] = []
    det_scores: List[float] = []
    det_kps: List[Dict[str, Tuple[float, float, float]]] = []

    if tiles_c * tiles_r <= 1:
        pose_res = backend.infer_pose(
            crop, conf=float(pose_conf), iou=float(pose_iou), max_det=int(pose_max_det), imgsz=int(pose_imgsz)
        )
        if pose_res.boxes is None or len(pose_res.boxes) == 0:
            return ([], [], [])
        pxy = pose_res.boxes.xyxy.cpu().numpy()
        psc = pose_res.boxes.conf.cpu().numpy() if getattr(pose_res.boxes, "conf", None) is not None else None
        for i in range(pxy.shape[0]):
            x1, y1, x2, y2 = [float(v) for v in pxy[i]]
            det_boxes.append((x1 + x, y1 + y, x2 + x, y2 + y))
            det_scores.append(float(psc[i]) if psc is not None else 1.0)
            det_kps.append(_kps_from_ultralytics(pose_res, i, offset_xy=(x, y)))
        return (det_boxes, det_scores, det_kps)

    tile_w = max(1, int(round(w / float(tiles_c))))
    tile_h = max(1, int(round(h / float(tiles_r))))
    # 轻微重叠，减少边界漏检（10%）
    ov_w = int(round(tile_w * 0.10))
    ov_h = int(round(tile_h * 0.10))
    for tr in range(tiles_r):
        for tc in range(tiles_c):
            tx1 = max(0, tc * tile_w - ov_w)
            ty1 = max(0, tr * tile_h - ov_h)
            tx2 = min(w, (tc + 1) * tile_w + ov_w)
            ty2 = min(h, (tr + 1) * tile_h + ov_h)
            tile = crop[ty1:ty2, tx1:tx2]
            if tile.size == 0:
                continue
            pose_res_t = backend.infer_pose(
                tile, conf=float(pose_conf), iou=float(pose_iou), max_det=int(pose_max_det), imgsz=int(pose_imgsz)
            )
            if pose_res_t.boxes is None or len(pose_res_t.boxes) == 0:
                continue
            pxy = pose_res_t.boxes.xyxy.cpu().numpy()
            psc = pose_res_t.boxes.conf.cpu().numpy() if getattr(pose_res_t.boxes, "conf", None) is not None else None
            off = (x + tx1, y + ty1)
            for i in range(pxy.shape[0]):
                x1, y1, x2, y2 = [float(v) for v in pxy[i]]
                det_boxes.append((x1 + off[0], y1 + off[1], x2 + off[0], y2 + off[1]))
                det_scores.append(float(psc[i]) if psc is not None else 1.0)
                det_kps.append(_kps_from_ultralytics(pose_res_t, i, offset_xy=off))
    return (det_boxes, det_scores, det_kps)


def _select_roi_interactive(frame_bgr: np.ndarray, max_disp_w: int = 1920, max_disp_h: int = 1080) -> Tuple[int, int, int, int]:
    """
    需要用户交互；如果环境不支持窗口，会抛异常。

    Windows 上常见问题：视频分辨率较大 + 系统缩放(125%/150%) 会导致 ROI 选择窗口只显示半张图。
    这里做“自动缩放显示”，并把用户框选的 ROI 坐标映射回原始分辨率。
    """
    title = "Select Target ROI"
    h0, w0 = frame_bgr.shape[:2]

    # 经验值：保证 ROI 选择窗口在大多数屏幕上完整显示（默认偏清晰）
    max_disp_w = int(max_disp_w) if int(max_disp_w) > 0 else 1920
    max_disp_h = int(max_disp_h) if int(max_disp_h) > 0 else 1080
    scale = min(max_disp_w / float(w0), max_disp_h / float(h0), 1.0)
    if scale < 1.0:
        # ROI 选择偏“看得清”：中等缩放用LINEAR更锐，超大缩放用AREA更稳
        interp = cv2.INTER_LINEAR if scale >= 0.6 else cv2.INTER_AREA
        disp = cv2.resize(frame_bgr, (int(round(w0 * scale)), int(round(h0 * scale))), interpolation=interp)
        print(f"[Task2] ROI选择窗口已自动缩放：{w0}x{h0} -> {disp.shape[1]}x{disp.shape[0]} (scale={scale:.3f})", flush=True)
    else:
        disp = frame_bgr

    # 窗口可缩放，避免被系统DPI裁切
    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    except cv2.error as e:
        msg = str(e)
        # 常见于安装了 opencv-python-headless（无 highgui），或某些环境缺少窗口后端
        if "The function is not implemented" in msg or "cvNamedWindow" in msg:
            raise RuntimeError(
                "当前 OpenCV 不包含 GUI/HighGUI 支持，无法使用 --t2-interactive-roi 弹窗框选。\n"
                "解决方案（二选一）：\n"
                "1) 安装带 GUI 的 OpenCV：先卸载 headless，再安装 opencv-python 或 opencv-contrib-python。\n"
                "2) 不用弹窗：改用 --t2-roi \"x,y,w,h\" 直接指定 ROI。"
            ) from e
        raise
    try:
        cv2.resizeWindow(title, int(disp.shape[1]), int(disp.shape[0]))
    except Exception:
        pass

    roi = cv2.selectROI(title, disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)

    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        raise ValueError("未选择有效ROI")

    # 映射回原图坐标
    if scale < 1.0:
        x = int(round(x / scale))
        y = int(round(y / scale))
        w = int(round(w / scale))
        h = int(round(h / scale))

    # clamp
    x = max(0, min(w0 - 1, x))
    y = max(0, min(h0 - 1, y))
    w = max(1, min(w0 - x, w))
    h = max(1, min(h0 - y, h))
    return x, y, w, h


def _tracker_create(name: str):
    n = (name or "CSRT").strip().upper()
    try:
        if n == "KCF":
            return cv2.TrackerKCF_create()
        return cv2.TrackerCSRT_create()
    except AttributeError:
        legacy = getattr(cv2, "legacy", None)
        if legacy is None:
            return None
        if n == "KCF" and hasattr(legacy, "TrackerKCF_create"):
            return legacy.TrackerKCF_create()
        if hasattr(legacy, "TrackerCSRT_create"):
            return legacy.TrackerCSRT_create()
        return None


def _iter_video_frames(cap: cv2.VideoCapture, start_frame: int, end_frame: int, log_prefix: str = "[Task2]"):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    idx = int(start_frame)
    while cap.isOpened() and idx < int(end_frame):
        ret, frame = cap.read()
        if not ret:
            print(f"{log_prefix} ⚠️ cap.read() 失败，idx={idx}，提前结束读取。", flush=True)
            break
        yield idx, frame
        idx += 1


def _load_seat_map_bbox(seat_map_path: str, student_id: str) -> Optional[Tuple[int, int, int, int]]:
    if not seat_map_path or not os.path.exists(seat_map_path):
        return None
    try:
        with open(seat_map_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        info = (obj.get("students") or {}).get(str(student_id))
        if not info:
            return None
        loc = info.get("estimated_location")
        if not loc or len(loc) != 4:
            return None
        t, r, b, l = [int(x) for x in loc]
        x, y, w, h = l, t, max(1, r - l), max(1, b - t)
        return x, y, w, h
    except Exception:
        return None


def _load_seat_map_all(seat_map_path: str) -> Dict[str, Dict]:
    """
    读取任务1生成的 seat_map.json，返回：
      sid -> {"name": str, "box_xyxy": (x1,y1,x2,y2)}
    estimated_location 存储格式为 [top, right, bottom, left]。
    """
    out: Dict[str, Dict] = {}
    if not seat_map_path or not os.path.exists(seat_map_path):
        return out
    try:
        with open(seat_map_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        students = obj.get("students") or {}
        for sid, info in students.items():
            if not isinstance(info, dict):
                continue
            loc = info.get("estimated_location")
            if not loc or len(loc) != 4:
                continue
            t, r, b, l = [int(x) for x in loc]
            x1, y1, x2, y2 = float(l), float(t), float(r), float(b)
            if x2 <= x1 or y2 <= y1:
                continue
            out[str(sid)] = {
                "name": str(info.get("name") or ""),
                "status": str(info.get("status") or ""),
                "box_xyxy": (x1, y1, x2, y2),
            }
        return out
    except Exception:
        return out


def _assign_to_seats_greedy(
    det_boxes: List[Tuple[float, float, float, float]],
    seat_boxes: Dict[str, Dict],
    min_iou: float,
) -> Dict[int, Dict]:
    """
    每帧将检测到的人框匹配到座位（student_id）。
    - 使用 IoU 贪心匹配：高 IoU 优先
    - 每个检测框最多匹配一个学生；每个学生最多匹配一个检测框
    返回：det_index -> {"student_id","student_name","seat_iou"}
    """
    if not det_boxes or not seat_boxes:
        return {}
    pairs: List[Tuple[float, int, str]] = []
    for di, b in enumerate(det_boxes):
        for sid, s in seat_boxes.items():
            sb = s.get("box_xyxy")
            if not sb:
                continue
            iou = float(_iou(b, tuple(map(float, sb))))
            if iou >= float(min_iou):
                pairs.append((iou, di, str(sid)))
    pairs.sort(key=lambda x: float(x[0]), reverse=True)
    used_det = set()
    used_sid = set()
    out: Dict[int, Dict] = {}
    for iou, di, sid in pairs:
        if di in used_det or sid in used_sid:
            continue
        used_det.add(di)
        used_sid.add(sid)
        info = seat_boxes.get(sid) or {}
        out[int(di)] = {"student_id": sid, "student_name": str(info.get("name") or ""), "seat_iou": float(iou)}
    return out


@dataclass
class Task2Config:
    video_path: str
    mode: str  # "person" | "class"
    start_minute: float
    duration_minutes: float
    sample_seconds: float  # default 10
    tracker: str  # CSRT/KCF
    roi: Optional[Tuple[int, int, int, int]]
    interactive_roi: bool
    output_dir: str
    output_json: str
    save_images: bool
    pose_model: str
    obj_model: str
    device: str  # auto/cuda/cpu
    imgsz: int  # 推理输入尺寸（越小越快）
    obj_every: int  # 物体检测频率：每 N 次采样跑一次
    no_obj: bool  # 仅做pose，不做物体检测（更快）
    obj_roi_imgsz: int  # person模式：桌面ROI二次物体检测输入尺寸（<=0 表示禁用）
    obj_roi_max_people: int  # class模式：每帧最多对多少个低头的人做ROI二次物体检测
    debug_kps: bool  # 关键点可见性诊断：在标记图上画关键点并输出置信度
    min_iou_keep: float  # 个人模式：防跳人门控，候选框与prev_box的最小IoU（严格=更大）
    max_shift_px: float  # 个人模式：防跳人门控，候选框中心与prev_box中心最大位移像素（严格=更小）
    relocal_ema_alpha: float  # 个人模式：重定位平滑系数（0~1，越大越跟随当前框，越小越平滑）
    obj_min_iou: float  # 物体归因：物体框与人框的最小IoU（严格=更大，减少误报）
    obj_min_conf: float  # 物体归因：物体置信度阈值（严格=更大，减少误报）
    roi_max_w: int  # 交互ROI窗口最大宽（用于避免只显示半张图/同时尽量清晰）
    roi_max_h: int  # 交互ROI窗口最大高
    seat_map_path: str
    student_id: Optional[str]
    images_max: int  # 保存标记图最大数量（<=0 表示不限制）
    images_every: int  # 每隔多少个采样保存1张（默认1=每次都保存）
    pose_conf: float  # pose 检测置信度阈值（越低越容易检出更多人，但误检可能上升）
    pose_iou: float  # pose NMS IoU（越高越“保留更多重叠框”）
    pose_max_det: int  # pose 最大检测人数上限
    pose_tiles: str  # class模式：pose 分块推理 cols,rows（如 "2,2"），默认 "1,1"
    pose_tile_dedupe_iou: float  # class模式：分块结果去重 IoU（越大越宽松），默认0.55
    class_roi_augment: bool  # class模式：若指定ROI，则额外做一次“全图pose”，与ROI分块结果合并（默认False）
    class_roi_imgsz: int  # class模式：ROI pass 专用 pose imgsz（>0生效），用于提升远处小人召回
    pose_min_upper_avg: float  # class模式：过滤假人框(书包/杂物)：上半身关键点平均置信度下限
    pose_min_upper_cnt: int  # class模式：过滤假人框：上半身关键点(conf>0)最少数量（不要求腿）
    pose_min_torso_cnt: int  # class模式：过滤假人框：躯干关键点(conf>0)最少数量
    pose_dedupe_center_thr: float  # class模式：追加去重：中心距离阈值比例(相对min(w,h))
    seat_assign_min_iou: float  # class模式：将检测框匹配到 seat_map 的最小 IoU 阈值（越大越严格）
    head_ema_alpha: float  # class模式：低头/向前看分数 EMA 平滑系数（0~1，越大越跟随当前）


class UltralyticsBackend:
    def __init__(self, pose_model: str, obj_model: str, device: str = "auto", imgsz: int = 640):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise ImportError(
                "未安装 ultralytics，无法使用 YOLOv8n-pose/RT-DETR-L。\n"
                "请先安装：python -m pip install ultralytics"
            ) from e

        # Ultralytics 的 device 参数不接受 "auto"，但我们 CLI 里为了好用提供了 auto/cuda/cpu。
        # 这里做兼容：auto -> (cuda可用则0，否则cpu)
        resolved_device = device
        try:
            import torch  # type: ignore

            d = (device or "").strip().lower()
            if d in {"", "auto"}:
                resolved_device = "0" if bool(torch.cuda.is_available()) else "cpu"
            elif d in {"cuda", "gpu"}:
                resolved_device = "0" if bool(torch.cuda.is_available()) else "cpu"
            elif d == "cpu":
                resolved_device = "cpu"
            else:
                # 允许用户传 "0" / "0,1" 等；若无CUDA则强制cpu避免报错
                if not bool(torch.cuda.is_available()) and d != "cpu":
                    resolved_device = "cpu"
        except Exception:
            # 极端情况下 torch 不可用，直接回退CPU
            resolved_device = "cpu"

        if str(resolved_device) != str(device):
            print(f"[Task2] device 纠正：{device} -> {resolved_device}", flush=True)

        self.imgsz = int(imgsz) if int(imgsz) > 0 else 640
        print(f"[Task2] 正在加载模型：pose={pose_model}, obj={obj_model}, device={resolved_device}, imgsz={self.imgsz}", flush=True)
        self._YOLO = YOLO
        try:
            self.pose = YOLO(pose_model)
        except RuntimeError as e:
            msg = str(e)
            if "PytorchStreamReader failed reading zip archive" in msg or "failed finding central directory" in msg:
                raise RuntimeError(
                    "加载 pose 权重失败：检测到权重文件可能损坏/下载不完整。\n"
                    f"- 当前 pose_model={pose_model}\n"
                    "建议：删除项目目录下同名 .pt 文件后重试，让 Ultralytics 重新下载。\n"
                    "例如（在 C:\\DP_PRO 下）：del yolov8s-pose.pt\n"
                    "如果网络受限，请改为手动下载官方权重并放到该路径。"
                ) from e
            raise

        try:
            self.obj = YOLO(obj_model)
        except RuntimeError as e:
            msg = str(e)
            if "PytorchStreamReader failed reading zip archive" in msg or "failed finding central directory" in msg:
                raise RuntimeError(
                    "加载 obj 权重失败：检测到权重文件可能损坏/下载不完整。\n"
                    f"- 当前 obj_model={obj_model}\n"
                    "建议：删除项目目录下同名 .pt 文件后重试，让 Ultralytics 重新下载。"
                ) from e
            raise
        self.device = resolved_device
        print("[Task2] 模型加载完成。", flush=True)

    def infer_pose(
        self,
        frame_bgr: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 100,
        imgsz: Optional[int] = None,
    ):
        return self.pose.predict(
            frame_bgr,
            device=self.device,
            imgsz=int(imgsz) if imgsz and int(imgsz) > 0 else self.imgsz,
            conf=float(conf),
            iou=float(iou),
            max_det=int(max_det),
            verbose=False,
        )[0]

    def infer_obj(self, frame_bgr: np.ndarray):
        return self.obj.predict(frame_bgr, device=self.device, imgsz=self.imgsz, verbose=False)[0]

    def infer_obj_with_imgsz(self, frame_bgr: np.ndarray, imgsz: int):
        """
        物体检测的可选高分辨率推理入口（用于 person 模式桌面ROI增强）。
        """
        return self.obj.predict(frame_bgr, device=self.device, imgsz=int(imgsz), verbose=False)[0]


def _obj_roi_from_person_box(
    person_box_xyxy: Tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    expand_x: float = 0.35,
    expand_y1: float = 0.10,
    expand_y2: float = 1.10,
) -> Tuple[int, int, int, int]:
    """
    生成“桌面相关”的 ROI，用于提高 laptop/book/phone 的召回（尤其是斜拍45°时电脑不在头正下方）。
    - 左右扩大：覆盖头的左下/右下桌面
    - 下方扩大：覆盖桌面区域
    - 上方略扩大：保留手部/上身
    """
    x1, y1, x2, y2 = [float(v) for v in person_box_xyxy]
    pw = max(1.0, x2 - x1)
    ph = max(1.0, y2 - y1)
    rx1 = x1 - float(expand_x) * pw
    ry1 = y1 - float(expand_y1) * ph
    rx2 = x2 + float(expand_x) * pw
    ry2 = y2 + float(expand_y2) * ph
    rx1_i = int(max(0, min(frame_w - 1, rx1)))
    ry1_i = int(max(0, min(frame_h - 1, ry1)))
    rx2_i = int(max(0, min(frame_w, rx2)))
    ry2_i = int(max(0, min(frame_h, ry2)))
    rw = max(1, int(rx2_i - rx1_i))
    rh = max(1, int(ry2_i - ry1_i))
    return (rx1_i, ry1_i, rw, rh)


COCO_KP_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def _kps_from_ultralytics(
    result_pose,
    idx_person: int,
    offset_xy: Optional[Tuple[int, int]] = None,
) -> Dict[str, Tuple[float, float, float]]:
    # result_pose.keypoints: (n,17,3)
    k = result_pose.keypoints
    if k is None:
        return {}
    arr = k.data.cpu().numpy()
    if idx_person >= arr.shape[0]:
        return {}
    ox, oy = (0, 0) if offset_xy is None else (int(offset_xy[0]), int(offset_xy[1]))
    out: Dict[str, Tuple[float, float, float]] = {}
    for i, name in enumerate(COCO_KP_NAMES):
        x, y, c = arr[idx_person, i, :]
        out[name] = (float(x) + ox, float(y) + oy, float(c))
    return out


def _boxes_from_ultralytics(result_obj, offset_xy: Optional[Tuple[int, int]] = None) -> List[DetBox]:
    boxes = []
    if result_obj.boxes is None:
        return boxes
    b = result_obj.boxes
    xyxy = b.xyxy.cpu().numpy()
    conf = b.conf.cpu().numpy()
    cls = b.cls.cpu().numpy().astype(int)
    names = getattr(result_obj, "names", {}) or {}
    ox, oy = (0, 0) if offset_xy is None else (int(offset_xy[0]), int(offset_xy[1]))
    for i in range(xyxy.shape[0]):
        name = names.get(int(cls[i]), str(int(cls[i])))
        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
        boxes.append(  # type: ignore
            DetBox(
                cls_name=str(name),
                conf=float(conf[i]),
                xyxy=(x1 + ox, y1 + oy, x2 + ox, y2 + oy),
            )
        )
    return boxes


def _draw_person(frame: np.ndarray, box_xyxy, label_lines: List[str]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    out = frame.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    y = max(0, y1 - 10)
    for line in label_lines[:6]:
        cv2.putText(out, line, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        y -= 16
        if y < 10:
            break
    return out


def _draw_kps_debug(
    frame: np.ndarray,
    kps: Dict[str, Tuple[float, float, float]],
    anchor_xy: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    关键点可见性诊断（更易读）：
    - 在关键点位置只画点（不在点旁边写字，避免“蓝色字体挤在一起”）
    - 在 anchor 附近画一个半透明信息面板，列出5个关键点的置信度
    """
    out = frame.copy()
    h, w = out.shape[:2]
    show = ["nose", "left_eye", "right_eye", "left_shoulder", "right_shoulder"]

    # 1) 画点
    for name in show:
        v = kps.get(name)
        if not v:
            continue
        x, y, c = v
        x_i, y_i = int(x), int(y)
        if x_i < 0 or y_i < 0 or x_i >= w or y_i >= h:
            continue
        # conf低于0.2用红色提示（与行为规则里的阈值保持一致）
        color = (0, 0, 255) if float(c) < 0.2 else (255, 0, 0)  # BGR
        cv2.circle(out, (x_i, y_i), 3, color, -1)

    # 2) 画面板（固定排版）
    # 为避免与绿色标签重叠：面板固定放在画面左侧边栏，y 跟随人框位置（或默认顶部）。
    ax, ay = (10, 10) if anchor_xy is None else anchor_xy
    ax = int(max(0, min(w - 1, ax)))
    ay = int(max(0, min(h - 1, ay)))

    lines: List[str] = []
    for name in show:
        v = kps.get(name)
        c = float(v[2]) if v else 0.0
        lines.append(f"{name}: {c:.2f}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    pad = 6
    line_h = 18
    panel_w = 190
    panel_h = pad * 2 + line_h * len(lines)

    # 面板固定在画面左侧
    x1 = 8
    # y 尽量居中对齐目标框（或默认顶部）
    y1 = ay - panel_h // 2
    y1 = int(max(0, min(h - panel_h - 1, y1)))
    x2 = min(w - 1, x1 + panel_w)
    y2 = min(h - 1, y1 + panel_h)

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)

    ty = y1 + pad + 14
    for i, line in enumerate(lines):
        name = show[i]
        v = kps.get(name)
        c = float(v[2]) if v else 0.0
        color = (0, 0, 255) if c < 0.2 else (255, 0, 0)
        cv2.putText(out, line, (x1 + pad, ty), font, font_scale, color, thickness, cv2.LINE_AA)
        ty += line_h
    return out


def run_task2(cfg: Task2Config) -> Dict:
    _safe_mkdir(cfg.output_dir)
    images_dir = os.path.join(cfg.output_dir, "images")
    if cfg.save_images:
        _safe_mkdir(images_dir)

    video_path = cfg.video_path
    print(
        f"[Task2] 启动：mode={cfg.mode}, start={cfg.start_minute}min, duration={cfg.duration_minutes}min, sample={cfg.sample_seconds}s",
        flush=True,
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(cfg.start_minute * 60 * fps) if fps > 0 else 0
    end_frame = int((cfg.start_minute + cfg.duration_minutes) * 60 * fps) if fps > 0 else total_frames
    end_frame = min(end_frame, total_frames)
    # 采样策略：
    # - sample_seconds <= 0：每一帧都处理（并可输出每一帧画面）
    # - sample_seconds > 0：按秒采样
    if float(cfg.sample_seconds) <= 0:
        sample_every = 1
    else:
        sample_every = int(max(1, round((fps if fps > 0 else 25.0) * float(cfg.sample_seconds))))

    expected = int(max(1, (end_frame - start_frame) / sample_every))
    print(
        f"[Task2] 视频：fps={fps:.2f}, total_frames={total_frames}, frame_range=[{start_frame},{end_frame}), sample_every={sample_every}帧, 预计采样≈{expected}次",
        flush=True,
    )

    try:
        backend = UltralyticsBackend(cfg.pose_model, cfg.obj_model, device=cfg.device, imgsz=int(cfg.imgsz))
    except ImportError as e:
        # 这里不要让异常“看起来像没执行”，直接在stdout给出可操作指引
        print("\n[Task2] ❌ 缺少依赖，无法执行 YOLO/RT-DETR 推理。", flush=True)
        print(f"[Task2] 详细原因: {e}", flush=True)
        print("[Task2] 请先安装：", flush=True)
        print("  python -m pip install ultralytics", flush=True)
        print("[Task2] 并确保已安装 torch（Ultralytics 依赖）。例如先装CPU版：", flush=True)
        print("  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu", flush=True)
        return {"output_json": None, "images_dir": None, "n": 0, "error": "missing_ultralytics_or_torch"}

    results: List[Dict] = []
    t0 = time.time()
    sample_count = 0
    last_log = 0

    try:
        if cfg.mode == "person":
            # 初始化 ROI
            roi = cfg.roi
            if roi is None and cfg.student_id:
                roi = _load_seat_map_bbox(cfg.seat_map_path, cfg.student_id)
            if roi is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                ret, frame0 = cap.read()
                if not ret:
                    raise ValueError("无法读取视频起始帧用于框选ROI")
                if not cfg.interactive_roi:
                    raise ValueError("个人模式需要 ROI：请使用 --t2-roi x,y,w,h 或 --t2-interactive-roi 或提供 --student-id 且 seat_map 存在")
                roi = _select_roi_interactive(frame0, max_disp_w=int(cfg.roi_max_w), max_disp_h=int(cfg.roi_max_h))
            print(f"[Task2] ROI 已确认：x,y,w,h={roi}", flush=True)

            tracker = _tracker_create(cfg.tracker)
            use_tracker = tracker is not None
            if not use_tracker:
                print("[Task2] ⚠️ 当前 OpenCV 未提供 CSRT/KCF 跟踪器，将降级为“每次采样用Pose按ROI重定位”模式。", flush=True)
                print("[Task2]    为避免漂移：将启用“位移/IoU门控 + 平滑”，超出阈值的候选框会被拒绝。", flush=True)

            # 初始化 tracker（如果可用）
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, frame0 = cap.read()
            if not ret:
                raise ValueError("无法读取视频起始帧")
            if use_tracker:
                tracker.init(frame0, roi)
                print(f"[Task2] 跟踪器已初始化：{cfg.tracker}", flush=True)

            # 注意：上面读过 1 帧，所以当前位置已到 start_frame+1
            frame_idx = start_frame + 1
            saved = 0
            images_max = int(cfg.images_max)
            images_every = max(1, int(cfg.images_every))
            # 防漂移参数（偏严格：更不容易“跳到后排同学”，代价是可能跳过某些采样帧）
            min_iou_keep = float(cfg.min_iou_keep)
            max_center_shift_px = float(cfg.max_shift_px)
            ema_alpha = float(cfg.relocal_ema_alpha)
            # 上一次“可信框”，用于约束下一次定位
            rx, ry, rw, rh = roi
            prev_box = (float(rx), float(ry), float(rx + rw), float(ry + rh))
            while cap.isOpened() and frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    print(f"[Task2] ⚠️ cap.read() 失败，frame_idx={frame_idx}，提前结束。", flush=True)
                    break

                person_box = None
                if use_tracker:
                    ok, box = tracker.update(frame)
                    if ok:
                        x, y, w, h = [int(v) for v in box]
                        person_box = (float(x), float(y), float(x + w), float(y + h))

                if (frame_idx - start_frame) % sample_every == 0:
                    sample_count += 1
                    if sample_count == 1 or (sample_count - last_log) >= 5:
                        elapsed = time.time() - t0
                        ts = float(frame_idx / fps) if fps else 0.0
                        print(f"[Task2] 进度(person)：{sample_count}/{expected} frame={frame_idx} t={ts:.1f}s elapsed={elapsed:.1f}s", flush=True)
                        last_log = sample_count
                    try:
                        t_pose0 = time.time()
                        pose_res = backend.infer_pose(frame, conf=float(cfg.pose_conf), iou=float(cfg.pose_iou), max_det=int(cfg.pose_max_det))
                        t_pose = time.time() - t_pose0
                    except Exception:
                        print("[Task2] ❌ 推理异常（person），已跳过该采样：", flush=True)
                        print(traceback.format_exc(), flush=True)
                        frame_idx += 1
                        continue
                    # obj：放到“选中目标人框”之后再跑，便于做桌面ROI高分辨率推理
                    run_obj = (not cfg.no_obj) and (int(cfg.obj_every) <= 1 or (sample_count % int(cfg.obj_every) == 0))
                    t_obj = 0.0
                    obj_boxes: List[DetBox] = []
                    chosen = None
                    if pose_res.boxes is not None and len(pose_res.boxes) > 0:
                        pxy = pose_res.boxes.xyxy.cpu().numpy()
                        best_iou = -1.0
                        if person_box is None:
                            rx, ry, rw, rh = roi
                            ref_box = (float(rx), float(ry), float(rx + rw), float(ry + rh))
                        else:
                            ref_box = person_box
                        for i in range(pxy.shape[0]):
                            iou = _iou(tuple(map(float, pxy[i])), ref_box)
                            if iou > best_iou:
                                best_iou = iou
                                chosen = i
                    if chosen is None:
                        frame_idx += 1
                        continue

                    pbox = tuple(map(float, pose_res.boxes.xyxy.cpu().numpy()[chosen]))
                    # 门控：候选框与 prev_box 的 IoU & 中心位移必须合理，否则拒绝（防止“跳到别的同学”）
                    def _center(b):
                        x1, y1, x2, y2 = b
                        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                    piou = float(_iou(pbox, prev_box))
                    pcx, pcy = _center(pbox)
                    ocx, ocy = _center(prev_box)
                    shift = ((pcx - ocx) ** 2 + (pcy - ocy) ** 2) ** 0.5
                    if (piou < min_iou_keep) and (shift > max_center_shift_px):
                        print(f"[Task2] ⚠️ 目标疑似漂移：reject iou={piou:.3f}, shift={shift:.1f}px (frame={frame_idx})", flush=True)
                        frame_idx += 1
                        continue

                    # EMA 平滑（减少抖动/微漂）
                    x1, y1, x2, y2 = pbox
                    px1, py1, px2, py2 = prev_box
                    smooth_box = (
                        px1 * (1 - ema_alpha) + x1 * ema_alpha,
                        py1 * (1 - ema_alpha) + y1 * ema_alpha,
                        px2 * (1 - ema_alpha) + x2 * ema_alpha,
                        py2 * (1 - ema_alpha) + y2 * ema_alpha,
                    )
                    person_box = smooth_box
                    prev_box = smooth_box
                    x1, y1, x2, y2 = [int(v) for v in pbox]
                    roi = (max(0, x1), max(0, y1), max(1, x2 - x1), max(1, y2 - y1))

                    kps = _kps_from_ultralytics(pose_res, chosen)
                    if run_obj and int(getattr(cfg, "obj_roi_imgsz", 0)) > 0:
                        try:
                            fh, fw = frame.shape[:2]
                            rx, ry, rw, rh = _obj_roi_from_person_box(person_box, frame_w=int(fw), frame_h=int(fh))
                            crop = frame[ry : ry + rh, rx : rx + rw]
                            t_obj0 = time.time()
                            obj_res_roi = backend.infer_obj_with_imgsz(crop, imgsz=int(getattr(cfg, "obj_roi_imgsz", 0)))
                            t_obj = time.time() - t_obj0
                            obj_boxes = _boxes_from_ultralytics(obj_res_roi, offset_xy=(rx, ry))
                        except Exception:
                            print("[Task2] ⚠️ 物体检测(ROI)异常，已忽略：", flush=True)
                            print(traceback.format_exc(), flush=True)
                            obj_boxes = []

                    if sample_count == 1 or sample_count % 5 == 0:
                        print(f"[Task2] 推理耗时(person)：pose={t_pose:.2f}s obj={t_obj:.2f}s (run_obj={run_obj})", flush=True)
                    beh = infer_behaviors(person_box, kps, obj_boxes, obj_min_iou=float(cfg.obj_min_iou), obj_min_conf=float(cfg.obj_min_conf))
                    results.append(
                        {
                            "frame_idx": int(frame_idx),
                            "timestamp_s": float(frame_idx / fps) if fps else 0.0,
                            "person_box": [float(v) for v in person_box],
                            "mode": "person",
                            "relocal": {"iou_prev": piou, "shift_px": shift},
                            "beh": beh,
                            "kps_conf": ({k: float(v[2]) for k, v in kps.items()} if cfg.debug_kps else None),
                        }
                    )

                    should_save = bool(cfg.save_images) and ((sample_count % images_every) == 0)
                    if should_save and (images_max <= 0 or saved < images_max):
                        label = [
                            f"head_down={beh['head_down']}",
                            f"forward={beh['looking_forward']}",
                            f"phone={beh['objects']['phone']}",
                            f"book={beh['objects']['book']}",
                            f"laptop={beh['objects']['laptop']}",
                        ]
                        out = _draw_person(frame, person_box, label)
                        if cfg.debug_kps:
                            out = _draw_kps_debug(out, kps, anchor_xy=(int(person_box[0]), int(person_box[1])))
                        cv2.imwrite(os.path.join(images_dir, f"frame_{int(frame_idx):06d}.jpg"), out)
                        saved += 1

                frame_idx += 1

        else:
            # 全班：每 sample_seconds 采样一帧
            # 支持 ROI（可用于“放大远处小人”）：对 ROI 区域裁剪后推理，并把坐标映射回原图
            saved = 0
            images_max = int(cfg.images_max)
            images_every = max(1, int(cfg.images_every))
            frames_read = 0

            class_roi = cfg.roi
            if class_roi is None and bool(cfg.interactive_roi):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
                ret, frame0 = cap.read()
                if not ret:
                    raise ValueError("无法读取视频起始帧用于框选ROI（class）")
                class_roi = _select_roi_interactive(frame0, max_disp_w=int(cfg.roi_max_w), max_disp_h=int(cfg.roi_max_h))
                print(f"[Task2] class ROI 已确认：x,y,w,h={class_roi}", flush=True)

            tiles_c, tiles_r = _parse_tiles(getattr(cfg, "pose_tiles", "1,1"))
            dedupe_iou = float(getattr(cfg, "pose_tile_dedupe_iou", 0.55))
            class_roi_augment = bool(getattr(cfg, "class_roi_augment", False))
            # 过滤“书包/杂物假人框”：只看上半身/躯干关键点，不要求腿（坐姿腿常不可见）
            pose_min_upper_avg = float(getattr(cfg, "pose_min_upper_avg", 0.10))
            pose_min_upper_cnt = int(getattr(cfg, "pose_min_upper_cnt", 2))
            pose_min_torso_cnt = int(getattr(cfg, "pose_min_torso_cnt", 1))
            # 去重补强：处理“同一人多框但IoU不高”的情况
            pose_dedupe_center_thr = float(getattr(cfg, "pose_dedupe_center_thr", 0.35))
            # seat_map 匹配：把检测到的人框映射到 student_id（用于输出按学生统计）
            seat_boxes = _load_seat_map_all(getattr(cfg, "seat_map_path", ""))
            seat_assign_min_iou = float(getattr(cfg, "seat_assign_min_iou", 0.05))
            head_ema_alpha = float(getattr(cfg, "head_ema_alpha", 0.35))
            # 每个 student_id 的时序平滑状态（score）
            head_state_by_sid: Dict[str, Dict[str, float]] = {}
            if seat_boxes:
                print(f"[Task2] seat_map 已加载：students={len(seat_boxes)}，assign_min_iou={seat_assign_min_iou}", flush=True)
                print(f"[Task2] head EMA 平滑启用：alpha={head_ema_alpha}", flush=True)
            if tiles_c * tiles_r > 1:
                print(f"[Task2] class pose 分块推理启用：tiles={tiles_c}x{tiles_r}, dedupe_iou={dedupe_iou}", flush=True)
            if class_roi is not None and class_roi_augment:
                print("[Task2] class ROI 增强启用：将执行“全图pose + ROI分块pose”并合并去重", flush=True)

            for frame_idx, frame in _iter_video_frames(cap, start_frame, end_frame, log_prefix="[Task2]"):
                frames_read += 1
                if (frame_idx - start_frame) % sample_every != 0:
                    continue
                sample_count += 1
                if sample_count == 1 or (sample_count - last_log) >= 3:
                    elapsed = time.time() - t0
                    ts = float(frame_idx / fps) if fps else 0.0
                    print(f"[Task2] 进度(class)：{sample_count}/{expected} frame={frame_idx} t={ts:.1f}s elapsed={elapsed:.1f}s", flush=True)
                    last_log = sample_count
                try:
                    # --- 准备ROI（class）
                    h0, w0 = frame.shape[:2]
                    roi_used: Tuple[int, int, int, int]
                    if class_roi is not None:
                        rx, ry, rw, rh = _clamp_roi(class_roi, w=w0, h=h0)
                        roi_used = (rx, ry, rw, rh)
                    else:
                        roi_used = (0, 0, w0, h0)

                    # --- Pose：全图 + (可选) ROI 分块增强
                    t_pose0 = time.time()
                    det_boxes: List[Tuple[float, float, float, float]] = []
                    det_scores: List[float] = []
                    det_kps: List[Dict[str, Tuple[float, float, float]]] = []

                    # 1) 全图 pass（避免“只框ROI内”）
                    if class_roi is not None and class_roi_augment:
                        b0, s0, k0 = _infer_pose_boxes_kps_for_region(
                            backend=backend,
                            frame_bgr=frame,
                            region_xywh=(0, 0, w0, h0),
                            tiles_c=1,
                            tiles_r=1,
                            pose_conf=float(cfg.pose_conf),
                            pose_iou=float(cfg.pose_iou),
                            pose_max_det=int(cfg.pose_max_det),
                            pose_imgsz=int(cfg.imgsz),
                        )
                        det_boxes.extend(b0)
                        det_scores.extend(s0)
                        det_kps.extend(k0)

                    # 2) ROI pass（可分块增强远处召回）
                    roi_imgsz = int(getattr(cfg, "class_roi_imgsz", 0))
                    if roi_imgsz <= 0:
                        roi_imgsz = int(cfg.imgsz)
                    b1, s1, k1 = _infer_pose_boxes_kps_for_region(
                        backend=backend,
                        frame_bgr=frame,
                        region_xywh=roi_used,
                        tiles_c=tiles_c,
                        tiles_r=tiles_r,
                        pose_conf=float(cfg.pose_conf),
                        pose_iou=float(cfg.pose_iou),
                        pose_max_det=int(cfg.pose_max_det),
                        pose_imgsz=int(roi_imgsz),
                    )
                    det_boxes.extend(b1)
                    det_scores.extend(s1)
                    det_kps.extend(k1)

                    t_pose = time.time() - t_pose0

                    # class 模式物体检测策略：
                    # - 若启用 obj_roi_imgsz>0：只对“低头”的人做桌面ROI二次检测（更准、但需要限额防止过慢）
                    # - 否则沿用原有：对全图/ROI 跑一次 obj，再给所有人做归因
                    obj_res = None
                    t_obj = 0.0
                    run_obj = (not cfg.no_obj) and (int(cfg.obj_every) <= 1 or (sample_count % int(cfg.obj_every) == 0))
                    obj_boxes_global: List[DetBox] = []
                    if run_obj and int(getattr(cfg, "obj_roi_imgsz", 0)) <= 0:
                        t_obj0 = time.time()
                        # 若启用 ROI 增强，为了给“ROI外的人”也能归因物体，物体检测改为全图
                        if class_roi is not None and class_roi_augment:
                            obj_res = backend.infer_obj(frame)
                            obj_boxes_global = _boxes_from_ultralytics(obj_res, offset_xy=None)
                        else:
                            rx, ry, rw, rh = roi_used
                            base_crop = frame[ry : ry + rh, rx : rx + rw]
                            obj_res = backend.infer_obj(base_crop)
                            obj_boxes_global = _boxes_from_ultralytics(obj_res, offset_xy=(rx, ry))
                        t_obj = time.time() - t_obj0
                except Exception:
                    print("[Task2] ❌ 推理异常（class），已跳过该采样：", flush=True)
                    print(traceback.format_exc(), flush=True)
                    continue
                if sample_count == 1 or sample_count % 3 == 0:
                    print(f"[Task2] 推理耗时(class)：pose={t_pose:.2f}s obj={t_obj:.2f}s (run_obj={run_obj})", flush=True)
                if not det_boxes:
                    continue

                # 1) 先过滤“假人框”（书包/杂物）
                det_boxes, det_scores, det_kps = _filter_pose_candidates(
                    det_boxes,
                    det_scores,
                    det_kps,
                    min_upper_avg=pose_min_upper_avg,
                    min_upper_cnt=pose_min_upper_cnt,
                    min_torso_cnt=pose_min_torso_cnt,
                )
                if not det_boxes:
                    continue

                # 2) NMS 去重（按IoU）
                keep_idx = _nms_xyxy(det_boxes, det_scores, iou_th=dedupe_iou) if len(det_boxes) > 1 else list(range(len(det_boxes)))
                det_boxes_kept = [det_boxes[i] for i in keep_idx]
                det_scores_kept = [det_scores[i] for i in keep_idx]
                det_kps_kept = [det_kps[i] for i in keep_idx]

                # 3) 中心距离去重（解决“一个人多框但IoU不高”）
                det_boxes_kept, det_scores_kept, det_kps_kept = _dedupe_by_center(
                    det_boxes_kept,
                    det_scores_kept,
                    det_kps_kept,
                    center_thr_ratio=pose_dedupe_center_thr,
                )

                # 4) 结合 seat_map 做 student_id 匹配（同一帧一座位只匹配一次）
                det_to_student = _assign_to_seats_greedy(det_boxes_kept, seat_boxes, min_iou=seat_assign_min_iou) if seat_boxes else {}

                # class 模式保存：每个采样帧只保存一张整帧图（避免每个人都保存导致爆炸）
                should_save_frame = bool(cfg.save_images) and ((sample_count % images_every) == 0) and (images_max <= 0 or saved < images_max)
                out_frame = frame.copy() if should_save_frame else None
                if should_save_frame and out_frame is not None and roi_used is not None:
                    rx, ry, rw, rh = roi_used
                    cv2.rectangle(out_frame, (int(rx), int(ry)), (int(rx + rw), int(ry + rh)), (0, 255, 255), 2)

                # 记录本帧“低头”的人，优先对其做 ROI 物体检测（限额）
                roi_imgsz_obj = int(getattr(cfg, "obj_roi_imgsz", 0))
                roi_obj_budget = int(getattr(cfg, "obj_roi_max_people", 15))
                roi_obj_budget = max(0, roi_obj_budget)
                roi_obj_used = 0

                for i in range(len(det_boxes_kept)):
                    person_box = tuple(map(float, det_boxes_kept[i]))
                    kps = det_kps_kept[i]
                    # 1) 先只用 pose 判头部状态
                    beh_head = infer_behaviors(person_box, kps, [], obj_min_iou=float(cfg.obj_min_iou), obj_min_conf=float(cfg.obj_min_conf))

                    # 2) 若低头且开启 ROI 物体检测：对桌面ROI做高分辨率二次检测
                    obj_boxes_person: List[DetBox] = []
                    if run_obj and roi_imgsz_obj > 0 and bool(beh_head.get("head_down")) and (roi_obj_used < roi_obj_budget):
                        try:
                            fh, fw = frame.shape[:2]
                            rx, ry, rw, rh = _obj_roi_from_person_box(person_box, frame_w=int(fw), frame_h=int(fh))
                            crop = frame[ry : ry + rh, rx : rx + rw]
                            obj_res_roi = backend.infer_obj_with_imgsz(crop, imgsz=int(roi_imgsz_obj))
                            obj_boxes_person = _boxes_from_ultralytics(obj_res_roi, offset_xy=(rx, ry))
                            roi_obj_used += 1
                        except Exception:
                            obj_boxes_person = []

                    # 3) 最终行为：优先使用 person ROI obj；否则回退到全局 obj（如果启用）
                    use_objs = obj_boxes_person if obj_boxes_person else obj_boxes_global
                    beh = infer_behaviors(person_box, kps, use_objs, obj_min_iou=float(cfg.obj_min_iou), obj_min_conf=float(cfg.obj_min_conf))
                    stu = det_to_student.get(int(i)) or {}

                    # --- head pose 分数时序平滑（按 student_id）
                    raw_head = (beh.get("head") or {}) if isinstance(beh, dict) else {}
                    raw_sd = float(raw_head.get("score_down") or 0.0)
                    raw_sf = float(raw_head.get("score_forward") or 0.0)
                    smooth_sd = raw_sd
                    smooth_sf = raw_sf
                    smooth_status = raw_head.get("status") or "unknown"
                    sid = str(stu.get("student_id")) if stu and stu.get("student_id") else None
                    if sid:
                        st = head_state_by_sid.setdefault(sid, {"sd": raw_sd, "sf": raw_sf})
                        a = float(head_ema_alpha)
                        st["sd"] = (1.0 - a) * float(st.get("sd", 0.0)) + a * raw_sd
                        st["sf"] = (1.0 - a) * float(st.get("sf", 0.0)) + a * raw_sf
                        smooth_sd = float(st["sd"])
                        smooth_sf = float(st["sf"])
                        # 三态：down/forward/unknown（用平滑后的分数）
                        if smooth_sd >= 0.60:
                            smooth_status = "down"
                        elif (smooth_sf >= 0.60) and (smooth_sd < 0.40):
                            smooth_status = "forward"
                        else:
                            smooth_status = "unknown"
                        beh["head"] = {
                            "status": smooth_status,
                            "score_down": smooth_sd,
                            "score_forward": smooth_sf,
                            "raw": {"status": raw_head.get("status"), "score_down": raw_sd, "score_forward": raw_sf},
                        }
                        beh["head_down"] = bool(smooth_status == "down")
                        beh["looking_forward"] = bool(smooth_status == "forward")
                    results.append(
                        {
                            "frame_idx": int(frame_idx),
                            "timestamp_s": float(frame_idx / fps) if fps else 0.0,
                            "person_box": [float(v) for v in person_box],
                            "mode": "class",
                            "person_index": int(i),
                            "student_id": (stu.get("student_id") if stu else None),
                            "student_name": (stu.get("student_name") if stu else None),
                            "seat_iou": (float(stu.get("seat_iou")) if stu and stu.get("seat_iou") is not None else None),
                            "beh": beh,
                            "kps_conf": ({k: float(v[2]) for k, v in kps.items()} if cfg.debug_kps else None),
                        }
                    )
                    if should_save_frame and out_frame is not None:
                        sid = str(stu.get("student_id")) if stu and stu.get("student_id") else None
                        sid_show = (sid[-4:] if sid and len(sid) >= 4 else sid) if sid else "NA"
                        label = [f"sid={sid_show}", f"down={beh['head_down']}", f"phone={beh['objects']['phone']}"]
                        out_frame = _draw_person(out_frame, person_box, label)
                        # debug_kps 在 class 模式很容易铺满画面，这里仅对第一个人做面板诊断
                        if cfg.debug_kps and i == 0:
                            out_frame = _draw_kps_debug(out_frame, kps, anchor_xy=(int(person_box[0]), int(person_box[1])))

                if should_save_frame and out_frame is not None:
                    cv2.imwrite(os.path.join(images_dir, f"frame_{int(frame_idx):06d}.jpg"), out_frame)
                    saved += 1

            print(f"[Task2] 读取结束：frames_read={frames_read}，samples={sample_count}/{expected}", flush=True)

    finally:
        try:
            cap.release()
        except Exception:
            pass

    print(f"[Task2] 结束：有效记录={len(results)}，耗时={time.time()-t0:.1f}s", flush=True)

    out_json_path = os.path.join(cfg.output_dir, cfg.output_json)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": cfg.mode,
                "start_minute": cfg.start_minute,
                "duration_minutes": cfg.duration_minutes,
                "sample_seconds": cfg.sample_seconds,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 额外输出：summary（全班总体 + 按学生ID统计）
    summary_path = os.path.join(cfg.output_dir, "task2_summary.json")
    try:
        if cfg.mode == "class":
            by_student: Dict[str, Dict] = {}
            by_frame: Dict[int, Dict] = {}
            by_minute: Dict[int, Dict] = {}

            def _inc(d: Dict, k: str, n: int = 1) -> None:
                d[k] = int(d.get(k, 0)) + int(n)

            for r in results:
                if r.get("mode") != "class":
                    continue
                frame_idx = int(r.get("frame_idx") or 0)
                ts = float(r.get("timestamp_s") or 0.0)
                minute = int(ts // 60)
                beh = r.get("beh") or {}
                objs = (beh.get("objects") or {}) if isinstance(beh, dict) else {}
                head_down = bool(beh.get("head_down"))
                forward = bool(beh.get("looking_forward"))
                phone = bool(objs.get("phone"))
                book = bool(objs.get("book"))
                laptop = bool(objs.get("laptop"))

                # frame-level
                fr = by_frame.setdefault(frame_idx, {"frame_idx": frame_idx, "timestamp_s": ts, "n": 0, "head_down": 0, "forward": 0, "phone": 0, "book": 0, "laptop": 0})
                _inc(fr, "n", 1)
                if head_down:
                    _inc(fr, "head_down", 1)
                if forward:
                    _inc(fr, "forward", 1)
                if phone:
                    _inc(fr, "phone", 1)
                if book:
                    _inc(fr, "book", 1)
                if laptop:
                    _inc(fr, "laptop", 1)

                # minute-level
                mn = by_minute.setdefault(minute, {"minute": minute, "n": 0, "head_down": 0, "forward": 0, "phone": 0, "book": 0, "laptop": 0})
                _inc(mn, "n", 1)
                if head_down:
                    _inc(mn, "head_down", 1)
                if forward:
                    _inc(mn, "forward", 1)
                if phone:
                    _inc(mn, "phone", 1)
                if book:
                    _inc(mn, "book", 1)
                if laptop:
                    _inc(mn, "laptop", 1)

                # student-level（需要 seat_map 匹配到 student_id）
                sid = r.get("student_id")
                if sid:
                    sid = str(sid)
                    st = by_student.setdefault(
                        sid,
                        {
                            "student_id": sid,
                            "student_name": r.get("student_name") or "",
                            "samples": 0,
                            "head_down": 0,
                            "forward": 0,
                            "phone": 0,
                            "book": 0,
                            "laptop": 0,
                        },
                    )
                    _inc(st, "samples", 1)
                    if head_down:
                        _inc(st, "head_down", 1)
                    if forward:
                        _inc(st, "forward", 1)
                    if phone:
                        _inc(st, "phone", 1)
                    if book:
                        _inc(st, "book", 1)
                    if laptop:
                        _inc(st, "laptop", 1)

            # ratios
            for st in by_student.values():
                n = max(1, int(st.get("samples", 0)))
                st["ratio"] = {
                    "head_down": float(st.get("head_down", 0)) / n,
                    "forward": float(st.get("forward", 0)) / n,
                    "phone": float(st.get("phone", 0)) / n,
                    "book": float(st.get("book", 0)) / n,
                    "laptop": float(st.get("laptop", 0)) / n,
                }
            for fr in by_frame.values():
                n = max(1, int(fr.get("n", 0)))
                fr["ratio"] = {k: float(fr.get(k, 0)) / n for k in ("head_down", "forward", "phone", "book", "laptop")}
            for mn in by_minute.values():
                n = max(1, int(mn.get("n", 0)))
                mn["ratio"] = {k: float(mn.get(k, 0)) / n for k in ("head_down", "forward", "phone", "book", "laptop")}

            summary = {
                "mode": cfg.mode,
                "start_minute": cfg.start_minute,
                "duration_minutes": cfg.duration_minutes,
                "sample_seconds": cfg.sample_seconds,
                "seat_map_path": cfg.seat_map_path,
                "seat_assign_min_iou": getattr(cfg, "seat_assign_min_iou", None),
                "overall_cn": {},
                "global": {
                    "frames": sorted(by_frame.values(), key=lambda x: int(x["frame_idx"])),
                    "minutes": sorted(by_minute.values(), key=lambda x: int(x["minute"])),
                },
                "by_student": sorted(by_student.values(), key=lambda x: (-int(x.get("samples", 0)), str(x.get("student_id", "")))),
                "notes": {
                    "mapping": "student_id 来自 seat_map 的座位框与检测框 IoU 贪心匹配；未匹配到的记录 student_id 为 null。",
                },
            }

            # --- 中文整体听课情况（比例）
            # 听课/抬头：looking_forward==True
            # 低头：head_down==True
            # unknown：两者都不是（EMA三态里可能出现）
            total_rec = 0
            n_forward = 0
            n_down = 0
            n_unknown = 0
            down_phone = 0
            down_laptop = 0
            down_book = 0
            down_other = 0
            for r in results:
                if r.get("mode") != "class":
                    continue
                beh = r.get("beh") or {}
                objs = (beh.get("objects") or {}) if isinstance(beh, dict) else {}
                bhs = (beh.get("behaviors") or {}) if isinstance(beh, dict) else {}
                hd = bool(beh.get("head_down"))
                fw = bool(beh.get("looking_forward"))
                total_rec += 1
                if hd:
                    n_down += 1
                    is_phone = bool(objs.get("phone")) or bool(bhs.get("playing_phone"))
                    is_laptop = bool(objs.get("laptop")) or bool(bhs.get("using_laptop"))
                    is_book = bool(objs.get("book")) or bool(bhs.get("note_taking"))
                    if is_phone:
                        down_phone += 1
                    elif is_laptop:
                        down_laptop += 1
                    elif is_book:
                        down_book += 1
                    else:
                        down_other += 1
                elif fw:
                    n_forward += 1
                else:
                    n_unknown += 1
            denom = max(1, int(total_rec))
            down_denom = max(1, int(n_down))
            summary["overall_cn"] = {
                "总记录数": int(total_rec),
                "听课(抬头/向前看)": {"数量": int(n_forward), "比例": float(n_forward) / denom},
                "低头": {"数量": int(n_down), "比例": float(n_down) / denom},
                "不确定": {"数量": int(n_unknown), "比例": float(n_unknown) / denom},
                "低头在做什么(按低头记录占比)": {
                    "看手机": float(down_phone) / down_denom,
                    "看电脑": float(down_laptop) / down_denom,
                    "看书/记笔记": float(down_book) / down_denom,
                    "其他/未知": float(down_other) / down_denom,
                },
                "物体检测策略": {
                    "说明": "若 --t2-obj-roi-imgsz>0，则仅对低头的人做桌面ROI二次物体检测（更准）；并限制每帧最多 --t2-obj-roi-max-people 人，避免过慢。",
                    "t2_obj_roi_imgsz": int(getattr(cfg, "obj_roi_imgsz", 0)),
                    "t2_obj_roi_max_people": int(getattr(cfg, "obj_roi_max_people", 0)),
                },
            }
            with open(summary_path, "w", encoding="utf-8") as sf:
                json.dump(summary, sf, ensure_ascii=False, indent=2)
        elif cfg.mode == "person":
            # person 模式：按时间比例统计 抬头/低头，以及“低头在做什么”
            # 说明：sample_seconds>0 时，每条记录视作代表一个采样窗口；
            # 若 sample_seconds<=0（逐帧），则用相邻 timestamp_s 的差作为近似时长。

            def _safe_float(x, default: float = 0.0) -> float:
                try:
                    return float(x)
                except Exception:
                    return float(default)

            rows = [r for r in results if (r.get("mode") == "person")]
            rows = sorted(rows, key=lambda rr: int(rr.get("frame_idx") or 0))

            if rows:
                # 计算每条记录的“代表时长”
                ts_list = [float(r.get("timestamp_s") or 0.0) for r in rows]
                deltas = [max(0.0, float(ts_list[i + 1] - ts_list[i])) for i in range(len(ts_list) - 1)]
                # 逐帧时用 median(delta) 作为典型间隔；采样模式直接用 sample_seconds
                if float(cfg.sample_seconds) > 0:
                    dt_default = float(cfg.sample_seconds)
                else:
                    d_sorted = sorted([d for d in deltas if d > 0.0])
                    dt_default = float(d_sorted[len(d_sorted) // 2]) if d_sorted else (1.0 / max(1.0, float(fps)))

                # last 用 dt_default
                dts = deltas + [dt_default]

                total_s = 0.0
                up_s = 0.0
                down_s = 0.0

                # 低头细分（时长）
                down_phone_s = 0.0
                down_laptop_s = 0.0
                down_book_s = 0.0
                down_other_s = 0.0

                # 可选：按分钟统计（用于画趋势）
                by_minute: Dict[int, Dict] = {}

                for r, dt in zip(rows, dts):
                    dt = float(max(0.0, dt))
                    beh = r.get("beh") or {}
                    head_down = bool(beh.get("head_down"))
                    forward = bool(beh.get("looking_forward"))
                    objs = (beh.get("objects") or {}) if isinstance(beh, dict) else {}
                    bhs = (beh.get("behaviors") or {}) if isinstance(beh, dict) else {}

                    total_s += dt
                    if head_down:
                        down_s += dt
                    else:
                        # 你的定义：只要不是低头就算抬头/向前看
                        up_s += dt

                    # 低头行为：优先 phone > laptop > book/note > other
                    if head_down:
                        is_phone = bool(objs.get("phone")) or bool(bhs.get("playing_phone"))
                        is_laptop = bool(objs.get("laptop")) or bool(bhs.get("using_laptop"))
                        is_book = bool(objs.get("book")) or bool(bhs.get("note_taking"))
                        if is_phone:
                            down_phone_s += dt
                        elif is_laptop:
                            down_laptop_s += dt
                        elif is_book:
                            down_book_s += dt
                        else:
                            down_other_s += dt

                    # minute aggregation
                    minute = int(float(r.get("timestamp_s") or 0.0) // 60)
                    m = by_minute.setdefault(
                        minute,
                        {
                            "分钟": int(minute),
                            "总时长秒": 0.0,
                            "抬头时长秒": 0.0,
                            "低头时长秒": 0.0,
                            "低头_看手机秒": 0.0,
                            "低头_看电脑秒": 0.0,
                            "低头_看书记笔记秒": 0.0,
                            "低头_其他秒": 0.0,
                        },
                    )
                    m["总时长秒"] = float(m["总时长秒"]) + dt
                    if head_down:
                        m["低头时长秒"] = float(m["低头时长秒"]) + dt
                        # 同步细分
                        is_phone = bool(objs.get("phone")) or bool(bhs.get("playing_phone"))
                        is_laptop = bool(objs.get("laptop")) or bool(bhs.get("using_laptop"))
                        is_book = bool(objs.get("book")) or bool(bhs.get("note_taking"))
                        if is_phone:
                            m["低头_看手机秒"] = float(m["低头_看手机秒"]) + dt
                        elif is_laptop:
                            m["低头_看电脑秒"] = float(m["低头_看电脑秒"]) + dt
                        elif is_book:
                            m["低头_看书记笔记秒"] = float(m["低头_看书记笔记秒"]) + dt
                        else:
                            m["低头_其他秒"] = float(m["低头_其他秒"]) + dt
                    else:
                        # forward 可能为 True/False（取决于 pose），但你定义里“非低头=抬头”
                        _ = forward
                        m["抬头时长秒"] = float(m["抬头时长秒"]) + dt

                denom = max(1e-6, float(total_s))
                down_denom = max(1e-6, float(down_s))

                summary_cn = {
                    "模式": "person",
                    "开始分钟": float(cfg.start_minute),
                    "时长分钟": float(cfg.duration_minutes),
                    "采样秒": float(cfg.sample_seconds),
                    "总样本数": int(len(rows)),
                    "总时长秒(估算)": float(total_s),
                    "抬头": {
                        "时长秒": float(up_s),
                        "比例": float(up_s / denom),
                        "说明": "按你的定义：只要不是低头就算抬头/向前看",
                    },
                    "低头": {
                        "时长秒": float(down_s),
                        "比例": float(down_s / denom),
                        "低头在做什么(按低头时长占比)": {
                            "看手机": float(down_phone_s / down_denom),
                            "看电脑": float(down_laptop_s / down_denom),
                            "看书/记笔记": float(down_book_s / down_denom),
                            "其他/未知": float(down_other_s / down_denom),
                        },
                        "低头在做什么(秒)": {
                            "看手机": float(down_phone_s),
                            "看电脑": float(down_laptop_s),
                            "看书/记笔记": float(down_book_s),
                            "其他/未知": float(down_other_s),
                        },
                    },
                    "按分钟趋势": sorted(by_minute.values(), key=lambda x: int(x.get("分钟", 0))),
                    "备注": {
                        "低头行为判定": "优先级：手机 > 电脑 > 书本/记笔记 > 其他；且已启用门控：非低头时 phone/laptop/book 强制为 false。",
                    },
                }

                with open(summary_path, "w", encoding="utf-8") as sf:
                    json.dump(summary_cn, sf, ensure_ascii=False, indent=2)
                # 控制台友好输出
                print(
                    "[Task2] person汇总："
                    f"抬头比例={summary_cn['抬头']['比例']:.2f}，低头比例={summary_cn['低头']['比例']:.2f}；"
                    f"低头：手机={summary_cn['低头']['低头在做什么(按低头时长占比)']['看手机']:.2f} "
                    f"电脑={summary_cn['低头']['低头在做什么(按低头时长占比)']['看电脑']:.2f} "
                    f"书/记笔记={summary_cn['低头']['低头在做什么(按低头时长占比)']['看书/记笔记']:.2f}",
                    flush=True,
                )
    except Exception:
        # summary 失败不影响主结果
        pass

    return {
        "output_json": out_json_path,
        "summary_json": (summary_path if os.path.exists(summary_path) else None),
        "images_dir": images_dir if cfg.save_images else None,
        "n": len(results),
    }


