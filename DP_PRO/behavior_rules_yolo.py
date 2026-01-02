"""
行为规则引擎（YOLO Pose + 物体检测）

目标：
- 输入：单个学生的姿态关键点（COCO 17点）+ 相关物体检测框（手机/电脑/书等）
- 输出：低头/向前看/记笔记/玩手机/使用电脑等标签

注意：规则是启发式的，需要结合机位与课堂场景调参。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math


@dataclass
class DetBox:
    cls_name: str
    conf: float
    xyxy: Tuple[float, float, float, float]  # (x1,y1,x2,y2)


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _kp(kps: Dict[str, Tuple[float, float, float]], name: str) -> Optional[Tuple[float, float]]:
    v = kps.get(name)
    if not v:
        return None
    x, y, c = v
    if c is None or float(c) < 0.2:
        return None
    return (float(x), float(y))


def infer_behaviors(
    person_box: Tuple[float, float, float, float],
    kps: Dict[str, Tuple[float, float, float]],
    objects: List[DetBox],
    obj_min_iou: float = 0.05,
    obj_min_conf: float = 0.35,
) -> Dict:
    """
    person_box: 人框(x1,y1,x2,y2)
    kps: 关键点字典 {name: (x,y,conf)}，COCO 17点
    objects: 物体检测结果（已过滤为关心的类别）
    """
    # ---- head_down / looking_forward（课堂远景启发式）
    nose = _kp(kps, "nose")
    l_sh = _kp(kps, "left_shoulder")
    r_sh = _kp(kps, "right_shoulder")
    l_eye = _kp(kps, "left_eye")
    r_eye = _kp(kps, "right_eye")
    l_hip = _kp(kps, "left_hip")
    r_hip = _kp(kps, "right_hip")

    def _clamp01(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))

    head_down = False
    looking_forward = False
    pose_ok = False
    head_status = "unknown"  # "down" | "forward"
    score_down = 0.0
    score_forward = 0.0
    if nose and l_sh and r_sh:
        pose_ok = True
        sh_mid = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
        # 尺度：优先肩-胯（更稳），没有胯则用肩宽兜底（避免用“肩-鼻”导致尺度自洽、判定钝化）
        if l_hip and r_hip:
            hip_mid = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)
            scale = max(1.0, _dist(sh_mid, hip_mid))
            sh_thr = 0.45  # 越小越严格（更不容易判低头）
        else:
            shoulder_w = max(1.0, _dist(l_sh, r_sh))
            scale = shoulder_w
            sh_thr = 0.30

        # 低头（斜视角/远景更鲁棒的启发式）：
        # 真实情况下“鼻子通常在肩膀上方”，低头时表现为“鼻子更接近肩线”（肩-鼻竖直距离变小），
        # 而不是鼻子跑到肩膀下方。
        delta_sh = sh_mid[1] - nose[1]  # >0 表示鼻子在肩膀上方
        norm_sh = float(delta_sh) / float(scale)  # 越小越“贴近肩线”
        head_down_sh = norm_sh < float(sh_thr)
        # 分数：norm_sh 越接近 0 越像低头；超过 sh_thr 基本不算低头
        score_down_sh = _clamp01((float(sh_thr) - float(norm_sh)) / max(1e-6, float(sh_thr)))

        head_down_eye = False
        score_down_eye = 0.0
        if l_eye and r_eye:
            eye_mid = ((l_eye[0] + r_eye[0]) / 2.0, (l_eye[1] + r_eye[1]) / 2.0)
            eye_dist = max(1.0, _dist(l_eye, r_eye))
            # 低头时鼻尖相对眼睛中心会明显下移
            norm_eye = float(nose[1] - eye_mid[1]) / float(eye_dist)
            head_down_eye = norm_eye > 0.55
            # 分数：>0.55 开始像低头，>0.95 基本确定（线性）
            score_down_eye = _clamp01((float(norm_eye) - 0.55) / 0.40)

        head_down = bool(head_down_sh or head_down_eye)
        score_down = float(max(score_down_sh, score_down_eye))

        # 你定义的 forward：只要“不是低头”就算 forward。
        # 因此 forward 的分数直接取 (1 - score_down)，并且不再输出 unknown（避免“既不down也不forward”）。
        score_forward = float(_clamp01(1.0 - float(score_down)))
        head_status = "down" if float(score_down) >= 0.60 else "forward"
        head_down = bool(head_status == "down")
        looking_forward = bool(head_status == "forward")
    else:
        # 关键点不足时：无法可靠判定低头。按你的定义，默认视作“不是低头 => forward”。
        pose_ok = False
        score_down = 0.0
        score_forward = 1.0
        head_status = "forward"
        head_down = False
        looking_forward = True

    # ---- objects: phone / laptop / book
    # 只取与“该学生相关区域”有关系的物体：
    # 坐姿场景里，书/电脑/手机常在桌面上，可能几乎不与人框重叠（尤其 person 模式 ROI 只框到头/上身时）。
    # 因此这里使用“向下扩展的人框”作为物体归因区域，并允许“物体中心落在区域内”视为相关。
    px1, py1, px2, py2 = [float(v) for v in person_box]
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)
    # 经验扩展：左右放宽、向下放宽以覆盖桌面区域（降低漏报）。
    # 说明：教室斜拍（约45°）时，电脑/书本往往不在“头正下方”，而在头的左下/右下。
    # 因此这里把“桌面区域”左右扩展更大，同时用“肩膀线以下”做门控，降低把邻座桌面算进来的概率。
    obj_region = (
        px1 - 0.35 * pw,
        py1 - 0.10 * ph,
        px2 + 0.35 * pw,
        py2 + 1.10 * ph,
    )

    def _center_in(box: Tuple[float, float, float, float], region: Tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = [float(v) for v in box]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        rx1, ry1, rx2, ry2 = [float(v) for v in region]
        return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)

    rel_objs = [
        o
        for o in objects
        if (_iou(o.xyxy, obj_region) >= float(obj_min_iou)) or _center_in(o.xyxy, obj_region)
    ]
    # 诊断用：保留相关物体列表（按置信度排序，避免 JSON 过大）
    rel_objs_sorted = sorted(rel_objs, key=lambda o: float(o.conf), reverse=True)[:12]

    # 当你为了“提升小目标召回”把 obj_min_conf 调得很低时，phone 类误检会显著上升。
    # 这里对 phone 做更严格的门槛，避免“没低头也总是识别手机”。
    phone_conf_thr = max(float(obj_min_conf), 0.35)

    def _area_xyxy(b: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = [float(v) for v in b]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    region_area = max(1.0, _area_xyxy(obj_region))

    def _phone_size_ok(b: Tuple[float, float, float, float]) -> bool:
        # phone 不应占据“人体+桌面归因区域”的太大比例；否则多半是桌面/书本/屏幕被误判成 phone
        return (_area_xyxy(b) / region_area) <= 0.18

    def _desk_gate_y() -> float:
        """桌面大致在肩膀线下方。没有肩膀点时回退到 person_box 的上 40% 位置。"""
        ls = _kp(kps, "left_shoulder")
        rs = _kp(kps, "right_shoulder")
        if ls and rs:
            return 0.5 * (float(ls[1]) + float(rs[1]))
        return float(py1 + 0.40 * ph)

    desk_y0 = _desk_gate_y()
    pcx = 0.5 * (px1 + px2)

    def _center_xy(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = [float(v) for v in b]
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def _desk_center_ok(b: Tuple[float, float, float, float]) -> bool:
        # 必须在肩膀线以下（桌面区域），并且不能离人框中心太远（避免邻座误归因）
        cx, cy = _center_xy(b)
        if cy <= (desk_y0 + 0.05 * ph):
            return False
        if abs(float(cx - pcx)) > (0.85 * pw):
            return False
        return True

    has_phone = any(
        o.cls_name in {"cell phone", "mobile phone", "phone"}
        and float(o.conf) >= phone_conf_thr
        and _phone_size_ok(o.xyxy)
        for o in rel_objs
    )

    # laptop/book：更依赖“桌面区域”归因（斜拍时常在头左下/右下），因此加桌面门控
    has_laptop = any(
        (o.cls_name == "laptop")
        and (float(o.conf) >= float(obj_min_conf))
        and _desk_center_ok(o.xyxy)
        for o in rel_objs
    )
    has_book = any(
        (o.cls_name == "book")
        and (float(o.conf) >= float(obj_min_conf))
        and _desk_center_ok(o.xyxy)
        for o in rel_objs
    )

    # 用户要求：只有在“低头”时，phone/laptop/book 才允许为 True；
    # 只要不是低头（你定义的 forward），这三项必须为 False（避免“抬头也识别到电脑/手机/书本”的误报）。
    if not bool(head_down):
        has_phone = False
        has_laptop = False
        has_book = False

    # 玩手机：手机存在 + 低头
    playing_phone = bool(has_phone and head_down)
    # 记笔记：书/本子存在 + 低头（倾向看桌面）
    note_taking = bool(has_book and head_down)
    # 使用电脑：laptop存在 + 低头
    using_laptop = bool(has_laptop and head_down)

    return {
        "pose_ok": pose_ok,
        "head_down": head_down,
        "looking_forward": looking_forward,
        "head": {
            "status": head_status,
            "score_down": float(score_down),
            "score_forward": float(score_forward),
        },
        "objects": {
            "phone": has_phone,
            "laptop": has_laptop,
            "book": has_book,
        },
        # 诊断字段：用于确认“模型是否检出book” vs “检出了但被阈值过滤/归因失败”
        "objects_debug": {
            "obj_region": [float(v) for v in obj_region],
            "phone_conf_thr": float(phone_conf_thr),
            "desk_y0": float(desk_y0),
            "gated_by_head_down": bool(not bool(head_down)),
            "related": [{"cls": o.cls_name, "conf": float(o.conf), "xyxy": [float(v) for v in o.xyxy]} for o in rel_objs_sorted],
        },
        "behaviors": {
            "playing_phone": playing_phone,
            "note_taking": note_taking,
            "using_laptop": using_laptop,
        },
    }


