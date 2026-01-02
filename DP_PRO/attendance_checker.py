"""
出勤检测模块
通过人脸识别分析没来上课的学生
"""
import cv2
import os
import sys
import numpy as np
import random
from typing import Dict, List, Set
from face_recognition_module import FaceRecognitionModule
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import json
import time


class AttendanceChecker:
    """出勤检测类"""
    
    def __init__(self, face_module: FaceRecognitionModule):
        """
        初始化出勤检测器
        
        Args:
            face_module: 人脸识别模块实例
        """
        self.face_module = face_module
        self.detected_students = set()  # 检测到的学生ID集合
        self.previous_frame_students = []  # 前一帧识别到的学生信息（用于跟踪）
        self.tracking_threshold = 50  # 跟踪阈值（像素距离），如果人脸位置变化小于此值，认为是同一人
        # 观测统计：用于减少“单帧误匹配”污染累计出勤
        # - hits: 该学生在多少个采样帧中被识别到（同一帧内会去重）
        # - best_confidence: 观测到的最高置信度（用于调参参考）
        self.student_observations: Dict[str, Dict[str, float]] = {}
        # 认为“已出勤”的最低命中帧数（避免单帧误识别）
        self.min_presence_hits = 3
        # 只有当单帧置信度达到该阈值时，才计入 hits（避免低质量匹配把人“刷进累计”）
        self.hit_confidence_threshold = 0.20
        # 用于最终“确认标签”的最低 best_confidence（进一步抑制持续误匹配带来的最终标注错误）
        self.min_confirm_best_confidence = 0.20
        # 位置样本：用于从多帧中估计每个学生的“稳定座位位置”
        # {student_id: [{"frame_idx": int, "confidence": float, "location": (t,r,b,l), "name": str}, ...]}
        self.student_location_samples: Dict[str, List[Dict]] = {}
    
    def check_attendance_from_video(
        self, 
        video_path: str, 
        sample_interval: int = 30,
        max_frames: int = 100,
        start_time_minutes: float = 0.0,
        output_images_dir: str = None
    ) -> Dict[str, Set[str]]:
        """
        从视频中检测出勤情况
        
        Args:
            video_path: 视频文件路径
            sample_interval: 采样间隔（帧数），每隔多少帧采样一次
            max_frames: 最大采样帧数
            start_time_minutes: 开始采样时间（分钟），从视频的哪个时间点开始采样
            output_images_dir: 输出图像目录，如果指定则保存带标记的图像
            
        Returns:
            包含检测结果的字典
        """
        # 创建输出目录
        if output_images_dir:
            os.makedirs(output_images_dir, exist_ok=True)
            print(f"将保存标记图像到: {output_images_dir}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps if fps > 0 else 0

        # 可选：写“任务1检测缓存”，供任务2复用（避免重复做人脸识别）
        cache_fp = None
        if output_images_dir:
            # 注意：缓存文件名固定，避免将外部输入直接用于文件路径（安全策略）。
            cache_path = os.path.join(output_images_dir, "detections_cache.jsonl")
            meta_path = os.path.join(output_images_dir, "detections_meta.json")
            try:
                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump(
                        {
                            "created_at": time.time(),
                            "video_basename": os.path.basename(video_path),
                            "fps": float(fps) if fps else 0.0,
                            "total_frames": int(total_frames),
                            "start_time_minutes": float(start_time_minutes),
                            "sample_interval": int(sample_interval),
                            "max_frames": int(max_frames),
                        },
                        mf,
                        ensure_ascii=False,
                        indent=2,
                    )
                cache_fp = open(cache_path, "w", encoding="utf-8")
                print(f"✅ 已启用任务1检测缓存输出: {cache_path}")
            except Exception as e:
                print(f"⚠️  警告: 无法写入任务1检测缓存，将跳过缓存输出。原因: {e}")
                cache_fp = None
        
        # 计算起始帧
        start_frame = int(start_time_minutes * 60 * fps) if fps > 0 else 0
        if start_frame >= total_frames:
            raise ValueError(f"开始时间 {start_time_minutes} 分钟超出视频长度（总时长约 {total_duration/60:.2f} 分钟）")
        
        # 跳转到起始帧
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            print(f"已跳转到第 {start_time_minutes:.2f} 分钟（帧 {start_frame}）")
        
        frame_count = start_frame
        sampled_count = 0
        frames_since_last_sample = 0  # 距离上次采样的帧数
        self.detected_students.clear()
        self.previous_frame_students = []  # 清空前一帧的识别结果
        self.student_observations.clear()
        self.student_location_samples.clear()
        last_processed_frame = None  # 最后一次“采样处理”的原始帧（用于调试/回退）
        last_processed_result = None  # 最后一次处理结果（含 face_locations / detected_students）
        last_processed_frame_idx = None
        # “随机汇总帧”：从采样帧中随机挑一帧作为可视化底图（水库采样，均匀随机）
        random_summary_frame = None
        random_summary_result = None
        random_summary_frame_idx = None
        
        print(f"开始分析视频: {video_path}")
        print(f"视频总时长: {total_duration/60:.2f} 分钟 ({total_frames} 帧, {fps:.2f} FPS)")
        print(f"采样开始时间: {start_time_minutes:.2f} 分钟")
        print(f"采样间隔: {sample_interval} 帧, 最大采样: {max_frames} 帧")
        
        # 跳转到开始时间
        if start_time_minutes > 0:
            start_frame = int(start_time_minutes * 60 * fps)
            print(f"正在跳转到第 {start_time_minutes:.2f} 分钟（帧 {start_frame}）...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            print(f"✅ 已跳转到指定位置")
        
        print("开始处理视频帧...")
        sys.stdout.flush()  # 确保输出立即显示
        
        while cap.isOpened() and sampled_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 每隔一定帧数采样一次（从开始时间后的第一帧开始采样）
            # 使用 frames_since_last_sample 确保从开始时间后立即采样第一帧
            if frames_since_last_sample >= sample_interval:
                # 显示正在处理的提示
                print(f"正在处理第 {sampled_count + 1} 个采样帧（当前视频帧: {frame_count}）...")
                sys.stdout.flush()
                
                # 处理当前帧
                result = self.face_module.process_video_frame(frame)
                
                # 使用跟踪机制改进识别结果
                tracked_result = self._track_students(result, frame)
                
                # 记录检测到的学生
                # 同一帧可能会出现同一学生被重复识别（例如两个相近人脸都匹配到同一学号），这里先按 student_id 去重
                # 并只保留该 student_id 下“最高置信度”的那条记录，避免统计与标注重复/污染累计。
                unique_students_by_id: Dict[str, Dict] = {}
                for student in tracked_result['detected_students']:
                    student_id = student.get('student_id')
                    if not student_id:
                        continue
                    prev = unique_students_by_id.get(student_id)
                    if prev is None or float(student.get('confidence', 0)) > float(prev.get('confidence', 0)):
                        unique_students_by_id[student_id] = student
                tracked_result['detected_students'] = list(unique_students_by_id.values())
                
                # 更新观测统计，并基于“命中帧数”确认出勤，减少单帧误识别
                for student_id, student in unique_students_by_id.items():
                    obs = self.student_observations.get(student_id)
                    conf = float(student.get('confidence', 0))
                    # 低于阈值的匹配不计入 hits（但仍可在当前帧显示/调试）
                    if conf < float(self.hit_confidence_threshold):
                        continue
                    # 记录位置样本（用于估计稳定座位位置）
                    loc = student.get("location")
                    if loc and len(loc) == 4:
                        try:
                            t, r, b, l = [int(x) for x in loc]
                            self.student_location_samples.setdefault(str(student_id), []).append(
                                {
                                    "frame_idx": int(frame_count),
                                    "confidence": float(conf),
                                    "location": (t, r, b, l),
                                    "name": str(student.get("name", "")),
                                }
                            )
                        except Exception:
                            pass
                    if obs is None:
                        self.student_observations[student_id] = {'hits': 1.0, 'best_confidence': conf}
                    else:
                        obs['hits'] = float(obs.get('hits', 0.0)) + 1.0
                        obs['best_confidence'] = max(float(obs.get('best_confidence', 0.0)), conf)
                    
                    if int(self.student_observations[student_id]['hits']) >= int(self.min_presence_hits):
                        self.detected_students.add(student_id)
                
                # 更新前一帧的识别结果
                self.previous_frame_students = tracked_result['detected_students'].copy()
                
                # 记录最后一次处理的帧与结果，用于结束时输出“最后一帧汇总标记图”
                last_processed_frame = frame.copy()
                last_processed_result = tracked_result
                last_processed_frame_idx = frame_count
                
                # 更新“随机汇总帧”（水库采样）：在第 k 个采样帧时，以 1/k 概率替换
                # 这样最终每个采样帧被选中的概率相同。
                k = int(sampled_count) + 1  # 当前是第 k 个采样帧（sampled_count 还未 +1）
                if k <= 1 or random.randrange(k) == 0:
                    random_summary_frame = frame.copy()
                    # tracked_result 里包含当前帧的 face_locations / detected_students，可用于画“参考帧标注”
                    random_summary_result = tracked_result
                    random_summary_frame_idx = frame_count
                
                # 如果指定了输出目录，保存带标记的图像
                if output_images_dir and len(tracked_result['detected_students']) > 0:
                    marked_frame = self._draw_student_labels(frame.copy(), tracked_result['detected_students'])
                    image_filename = f"frame_{sampled_count:03d}_t{frame_count}.jpg"
                    image_path = os.path.join(output_images_dir, image_filename)
                    cv2.imwrite(image_path, marked_frame)

                # 写入结构化缓存（每条记录一行 JSON，方便流式读取）
                if cache_fp is not None and tracked_result.get("detected_students"):
                    for s in tracked_result["detected_students"]:
                        try:
                            sid = str(s.get("student_id", "")).strip()
                            loc = s.get("location")
                            if not sid or not loc or len(loc) != 4:
                                continue
                            cache_fp.write(
                                json.dumps(
                                    {
                                        "frame_idx": int(frame_count),
                                        "timestamp_s": float(frame_count / fps) if fps else 0.0,
                                        "student_id": sid,
                                        "name": str(s.get("name", "")),
                                        "confidence": float(s.get("confidence", 0.0)),
                                        "location": [int(loc[0]), int(loc[1]), int(loc[2]), int(loc[3])],  # top,right,bottom,left
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        except Exception:
                            # 缓存写失败不影响主流程
                            pass
                
                sampled_count += 1
                frames_since_last_sample = 0  # 重置计数器
                # 每处理一个采样帧都显示进度
                print(f"\n[{sampled_count}/{max_frames}] ✅ 完成采样帧 {sampled_count}")
                print(f"  当前视频帧: {frame_count}, 检测到 {result['total_faces']} 个人脸，匹配到 {len(tracked_result['detected_students'])} 个学生")
                print(f"  累计检测到学生数（确认出勤：hits≥{self.min_presence_hits} 且 单帧置信度≥{self.hit_confidence_threshold:.2f}）: {len(self.detected_students)}")
                print(f"  疑似识别到学生数（满足单帧置信度≥{self.hit_confidence_threshold:.2f} 的命中过≥1帧）: {len(self.student_observations)}")
                # 显示匹配置信度信息
                if tracked_result['detected_students']:
                    confidences = [s.get('confidence', 0) for s in tracked_result['detected_students']]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    min_confidence = min(confidences) if confidences else 0
                    print(f"  匹配置信度: 平均 {avg_confidence:.3f}, 最低 {min_confidence:.3f}")
                    # 显示检测到的学生姓名
                    student_names = [s.get('name', s.get('student_id', '未知')) for s in tracked_result['detected_students']]
                    print(f"  检测到的学生: {', '.join(student_names[:5])}{'...' if len(student_names) > 5 else ''}")
                sys.stdout.flush()  # 确保输出立即显示
            
            frame_count += 1
            frames_since_last_sample += 1
        
        cap.release()
        if cache_fp is not None:
            try:
                cache_fp.close()
            except Exception:
                pass
        
        print(f"\n{'='*60}")
        print(f"✅ 视频分析完成！")
        print(f"{'='*60}")
        print(f"共处理 {sampled_count} 个采样帧")
        print(f"检测到 {len(self.detected_students)} 个学生")
        sys.stdout.flush()
        
        if len(self.detected_students) == 0:
            print("\n⚠️  警告: 未检测到任何学生！")
            print("可能的原因:")
            print("  1. 视频中的人脸太小或模糊")
            print(f"  2. 人脸匹配阈值/规则导致未达到“确认出勤”条件（当前 tolerance={getattr(self.face_module, 'tolerance', '未知')}，hits≥{self.min_presence_hits}，单帧置信度阈值≥{self.hit_confidence_threshold:.2f}，best_confidence≥{self.min_confirm_best_confidence:.2f}）")
            print("  3. 照片质量与视频中的人脸差异太大")
            print("  4. 视频中的人脸角度或光照条件与照片差异较大")
            print(f"  5. 多数匹配置信度低于当前单帧命中阈值（{self.hit_confidence_threshold:.2f}）或未在多帧中重复命中")
            print("\n建议:")
            print("  1. 检查照片质量，确保人脸清晰、正面")
            print("  2. 如果需要更宽松的匹配，可以提高 --tolerance，或降低 hits/置信度门槛（但误识别会增加）")
            print("  2. 检查视频质量，确保人脸清晰可见")
            print("  3. 检查学生照片是否成功加载（应该看到 '已加载: 姓名 (学号)' 的提示）")
        
        if output_images_dir:
            print(f"已保存 {sampled_count} 张标记图像到 {output_images_dir}")
            # 额外保存两张“总结图”：
            # 1) 随机汇总帧：从采样帧中随机挑一帧作为底图
            # 2) 总结面板图：纯文字汇总全程结果（不依赖某一帧是否拍到所有人）
            chosen_frame = random_summary_frame if random_summary_frame is not None else last_processed_frame
            chosen_result = random_summary_result if random_summary_result is not None else last_processed_result
            chosen_idx = random_summary_frame_idx if random_summary_frame_idx is not None else last_processed_frame_idx

            # 最终总结图（真正的“几十帧总结后结果”）：面板 + 参考帧合成
            if chosen_frame is not None and chosen_result is not None:
                final_img = self._create_final_summary_image(
                    chosen_frame.copy(),
                    chosen_result.get('face_locations', []),
                    chosen_result.get('detected_students', []),
                )
                suffix = f"_t{chosen_idx}" if chosen_idx is not None else ""
                final_path = os.path.join(output_images_dir, f"final_summary{suffix}.jpg")
                if final_img is not None:
                    cv2.imwrite(final_path, final_img)
                    print(f"已保存最终总结图（全程统计 + 参考帧）到: {final_path}")

                # 仍然单独保留一张“参考帧标注图”，方便你放大看框细节
                ref = self._draw_all_faces_labels(
                    chosen_frame.copy(),
                    chosen_result.get('face_locations', []),
                    chosen_result.get('detected_students', []),
                )
                ref_path = os.path.join(output_images_dir, f"summary_reference_frame{suffix}.jpg")
                cv2.imwrite(ref_path, ref)
                print(f"已保存参考帧标注图到: {ref_path}")

            # 生成“稳定座位位置地图”（供任务2/3直接定位）
            # 说明：对每个学生把多帧 location 按中心点分桶聚类，选“命中次数最多”的簇（频次最大），
            # 再用该簇的中位数框作为该学生的最终唯一位置（一个人只输出一个位置）。
            seat_map = {}

            def _center(loc):
                t, r, b, l = loc
                return ((l + r) // 2, (t + b) // 2)

            def _iou_trbl(a, b) -> float:
                at, ar, ab, al = [int(x) for x in a]
                bt, br, bb, bl = [int(x) for x in b]
                ax1, ay1, ax2, ay2 = al, at, ar, ab
                bx1, by1, bx2, by2 = bl, bt, br, bb
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = float(iw * ih)
                if inter <= 0:
                    return 0.0
                area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
                area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
                denom = area_a + area_b - inter
                return float(inter / denom) if denom > 0 else 0.0

            for sid, samples in (self.student_location_samples or {}).items():
                if not samples:
                    continue
                locs = [s["location"] for s in samples if s.get("location")]
                if not locs:
                    continue
                # 分桶聚类（按 80px 网格）
                bin_size = 80
                bins = {}
                for loc in locs:
                    cx, cy = _center(loc)
                    key = (int(cx) // bin_size, int(cy) // bin_size)
                    bins[key] = bins.get(key, 0) + 1
                best_key = max(bins.items(), key=lambda kv: kv[1])[0]
                anchor_cx = best_key[0] * bin_size + bin_size // 2
                anchor_cy = best_key[1] * bin_size + bin_size // 2
                max_dist2 = (2 * bin_size) * (2 * bin_size)
                kept = []
                kept_conf = []
                for s in samples:
                    loc = s.get("location")
                    if not loc:
                        continue
                    cx, cy = _center(loc)
                    d2 = (cx - anchor_cx) * (cx - anchor_cx) + (cy - anchor_cy) * (cy - anchor_cy)
                    if d2 <= max_dist2:
                        kept.append(loc)
                        kept_conf.append(float(s.get("confidence", 0.0)))
                if not kept:
                    continue
                arr = np.array(kept, dtype=np.int32)
                med = np.median(arr, axis=0)
                est_loc = [int(med[0]), int(med[1]), int(med[2]), int(med[3])]  # t,r,b,l
                obs = self.student_observations.get(str(sid), {}) or {}
                hits = int(float(obs.get("hits", 0.0))) if obs else 0
                best_conf = float(obs.get("best_confidence", 0.0)) if obs else 0.0
                support = float(len(kept)) / float(len(samples))
                status = "confirmed" if (str(sid) in self.detected_students) else "suspected"
                if support < 0.6:
                    status = "ambiguous"
                # 取最近一次的 name（不依赖 student_list 参数）
                name = ""
                for s in reversed(samples):
                    n = str(s.get("name", "")).strip()
                    if n:
                        name = n
                        break
                seat_map[str(sid)] = {
                    "student_id": str(sid),
                    "name": name,
                    "status": status,
                    "estimated_location": est_loc,
                    "samples_total": int(len(samples)),
                    "samples_kept": int(len(kept)),
                    "cluster_support": support,
                    "hits": hits,
                    "best_confidence": best_conf,
                    "avg_confidence_kept": float(np.mean(kept_conf)) if kept_conf else 0.0,
                }

            # --- 位置冲突消解：避免同一位置出现多个学生标注（表现为同一处画两个框/两个标签）
            # 原因通常是误识别导致两个 student_id 的位置簇挤到同一个座位。
            # 策略：按“证据强度”排序（confirmed>suspected>ambiguous, hits, best_confidence, cluster_support），
            # 然后做贪心去重：如果与已保留的位置 IoU 很高（或落在同一网格），则丢弃较弱者（不画框）。
            if seat_map:
                bin_size = 80
                iou_thr = 0.55
                items = list(seat_map.items())

                def _status_rank(s: str) -> int:
                    ss = (s or "").lower()
                    if ss == "confirmed":
                        return 3
                    if ss == "suspected":
                        return 2
                    return 1  # ambiguous/unknown

                def _strength(info: Dict) -> tuple:
                    return (
                        _status_rank(str(info.get("status", ""))),
                        int(info.get("hits", 0)),
                        float(info.get("best_confidence", 0.0)),
                        float(info.get("cluster_support", 0.0)),
                        float(info.get("avg_confidence_kept", 0.0)),
                    )

                items.sort(key=lambda kv: _strength(kv[1]), reverse=True)
                kept: List[tuple] = []  # [(sid, info)]
                used_bins = set()
                # 回退：不再额外输出/解释冲突丢弃名单，仅用于内部去重
                for sid, info in items:
                    loc = info.get("estimated_location")
                    if not loc or len(loc) != 4:
                        continue
                    cx, cy = _center(loc)
                    key = (int(cx) // bin_size, int(cy) // bin_size)
                    conflict = False
                    # 先按网格粗去重
                    if key in used_bins:
                        conflict = True
                    else:
                        # 再按 IoU 精去重
                        for ks, ki in kept:
                            kloc = ki.get("estimated_location")
                            if kloc and float(_iou_trbl(loc, kloc)) >= float(iou_thr):
                                conflict = True
                                break
                    if conflict:
                        continue
                    kept.append((sid, info))
                    used_bins.add(key)

                seat_map = {sid: info for sid, info in kept}

            if seat_map:
                seat_map_path = os.path.join(output_images_dir, "seat_map.json")
                try:
                    with open(seat_map_path, "w", encoding="utf-8") as sf:
                        json.dump(
                            {
                                "video_basename": os.path.basename(video_path),
                                "fps": float(fps) if fps else 0.0,
                                "start_time_minutes": float(start_time_minutes),
                                "sample_interval": int(sample_interval),
                                "max_frames": int(max_frames),
                                "min_presence_hits": int(self.min_presence_hits),
                                "hit_confidence_threshold": float(self.hit_confidence_threshold),
                                "min_confirm_best_confidence": float(self.min_confirm_best_confidence),
                                "students": seat_map,
                            },
                            sf,
                            ensure_ascii=False,
                            indent=2,
                        )
                    print(f"✅ 已保存座位位置地图: {seat_map_path}")
                except Exception as e:
                    print(f"⚠️  警告: 无法写入 seat_map.json: {e}")

                # 同时输出一张可视化图片，方便你人工核对
                if chosen_frame is not None:
                    try:
                        seat_students = []
                        for sid, info in seat_map.items():
                            loc = info.get("estimated_location")
                            if not loc or len(loc) != 4:
                                continue
                            seat_students.append(
                                {
                                    "student_id": sid,
                                    "name": info.get("name", ""),
                                    "confidence": float(info.get("best_confidence", 0.0)),
                                    "location": tuple(int(x) for x in loc),
                                }
                            )
                        seat_img = self._draw_student_labels(chosen_frame.copy(), seat_students)
                        seat_img_path = os.path.join(output_images_dir, f"seat_map{suffix}.jpg")
                        cv2.imwrite(seat_img_path, seat_img)
                        print(f"✅ 已保存座位位置可视化图: {seat_img_path}")
                    except Exception as e:
                        print(f"⚠️  警告: 无法生成 seat_map.jpg: {e}")
        
        return {
            'detected_students': self.detected_students,
            'total_frames_processed': sampled_count
        }
    
    def _track_students(self, current_result: Dict, frame) -> Dict:
        """
        基于前一帧的识别结果跟踪当前帧的学生
        
        Args:
            current_result: 当前帧的识别结果
            frame: 当前帧图像
            
        Returns:
            改进后的识别结果（包含跟踪识别到的学生）
        """
        tracked_students = current_result['detected_students'].copy()
        current_face_locations = current_result['face_locations']
        
        # 如果没有前一帧数据，直接返回当前结果
        if len(self.previous_frame_students) == 0:
            return current_result
        
        # 对于当前帧检测到的人脸，如果位置与前一帧某个已识别学生接近，使用前一帧的识别结果
        # 这样可以保持识别连续性，减少错乱
        used_previous_indices = set()
        
        # 为每个当前帧检测到但未识别的人脸，尝试匹配前一帧的识别结果
        for i, face_loc in enumerate(current_face_locations):
            top, right, bottom, left = face_loc
            current_center = ((left + right) // 2, (top + bottom) // 2)
            
            # 检查当前帧是否已经识别到这个位置的学生
            already_matched = False
            for tracked in tracked_students:
                # 检查位置是否相同（允许小的误差）
                tracked_loc = tracked['location']
                if abs(tracked_loc[0] - top) < 5 and abs(tracked_loc[1] - right) < 5 and \
                   abs(tracked_loc[2] - bottom) < 5 and abs(tracked_loc[3] - left) < 5:
                    already_matched = True
                    break
            
            # 如果当前帧已经识别到，保留结果
            if already_matched:
                continue
            
            # 在前一帧的识别结果中查找位置接近的学生
            best_match = None
            best_distance = float('inf')
            best_prev_idx = -1
            
            for prev_idx, prev_student in enumerate(self.previous_frame_students):
                if prev_idx in used_previous_indices:
                    continue
                
                prev_top, prev_right, prev_bottom, prev_left = prev_student['location']
                prev_center = ((prev_left + prev_right) // 2, (prev_top + prev_bottom) // 2)
                
                # 计算中心点距离
                distance = np.sqrt((current_center[0] - prev_center[0])**2 + 
                                  (current_center[1] - prev_center[1])**2)
                
                if distance < self.tracking_threshold and distance < best_distance:
                    best_match = prev_student
                    best_distance = distance
                    best_prev_idx = prev_idx
            
            # 如果找到匹配，使用前一帧的识别结果
            if best_match is not None:
                tracked_student = {
                    'student_id': best_match['student_id'],
                    'name': best_match['name'],
                    'confidence': best_match.get('confidence', 0.85),  # 跟踪识别给较高置信度
                    'location': face_loc
                }
                tracked_students.append(tracked_student)
                used_previous_indices.add(best_prev_idx)
        
        return {
            'detected_students': tracked_students,
            'face_locations': current_face_locations,
            'total_faces': len(current_face_locations)
        }
    
    def _draw_student_labels(self, frame, detected_students: List[Dict]):
        """
        在图像上绘制学生标签（姓名和学号）- 支持中文
        
        Args:
            frame: 视频帧 (BGR格式)
            detected_students: 检测到的学生列表
            
        Returns:
            标记后的图像 (BGR格式)
        """
        if len(detected_students) == 0:
            return frame
        
        # 将OpenCV的BGR格式转换为RGB格式（PIL使用RGB）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # 尝试加载中文字体
        font_size = 20
        try:
            # Windows系统常见中文字体路径
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",  # 黑体
                "C:/Windows/Fonts/simsun.ttc",  # 宋体
                "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
                "C:/Windows/Fonts/simkai.ttf",  # 楷体
            ]
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except:
                        continue
            
            # 如果找不到字体，使用默认字体（可能不支持中文）
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # 先绘制所有人脸框（使用OpenCV，因为PIL绘制矩形不够灵活）
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        def _status_for_student_id(student_id: str) -> str:
            """返回 'confirmed' | 'suspected' | 'unknown'，基于全程统计。"""
            if not student_id:
                return "unknown"
            obs = self.student_observations.get(str(student_id))
            best_conf = float(obs.get('best_confidence', 0.0)) if isinstance(obs, dict) else 0.0
            confirmed = (str(student_id) in self.detected_students) and (best_conf >= float(self.min_confirm_best_confidence))
            return "confirmed" if confirmed else "suspected"

        for student in detected_students:
            top, right, bottom, left = student['location']
            sid = str(student.get("student_id", "")).strip()
            status = _status_for_student_id(sid)
            if status == "confirmed":
                color = (0, 255, 0)      # BGR 绿色：确认
            elif status == "suspected":
                color = (255, 0, 0)      # BGR 蓝色：疑似
            else:
                color = (0, 255, 255)    # BGR 黄色：未知
            cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)
        
        # 转换回PIL格式绘制文本
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # 绘制所有文本标签
        for student in detected_students:
            top, right, bottom, left = student['location']
            
            # 准备标签文本
            name = student['name']
            student_id = student['student_id']
            # 清理学号格式（去掉.0）
            student_id_clean = str(student_id).rstrip('.0')
            # 学号只显示“前2位+末2位”，例如 25263050019 -> 2519
            digits_only = ''.join([c for c in student_id_clean if c.isdigit()])
            short_id = student_id_clean
            if len(digits_only) >= 4:
                short_id = f"{digits_only[:2]}{digits_only[-2:]}"
            label = f"{name}{short_id}"
            confidence = student.get('confidence', 0)

            sid = str(student.get("student_id", "")).strip()
            status = _status_for_student_id(sid)
            obs = self.student_observations.get(str(sid)) if sid else None
            hits = int(obs.get("hits", 0)) if isinstance(obs, dict) else 0
            # 在标注图上明确区分“确认/疑似”，并显示 hits 便于理解为何未确认
            if status == "confirmed":
                label_with_conf = f"{label} {float(confidence):.2f}"
                bg_fill = (0, 255, 0)    # 绿色背景：确认
            elif status == "suspected":
                label_with_conf = f"疑似:{label} {float(confidence):.2f} h={hits}"
                bg_fill = (0, 0, 255)    # 蓝色背景：疑似（PIL RGB）
            else:
                label_with_conf = f"未知 {float(confidence):.2f}"
                bg_fill = (255, 255, 0)  # 黄色背景：未知（PIL RGB）
            
            # 使用PIL绘制文本（支持中文）
            # 先计算文本大小
            bbox = draw.textbbox((0, 0), label_with_conf, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 确定标签位置（尽量在人脸框上方，如果上方空间不够则放在下方）
            label_y = top - text_height - 5 if top - text_height - 5 > 0 else bottom + 5
            
            # 绘制文本背景（按确认/疑似/未知着色）
            bg_left = left
            bg_top = label_y - 2
            bg_right = left + text_width + 4
            bg_bottom = label_y + text_height + 2
            draw.rectangle([bg_left, bg_top, bg_right, bg_bottom], fill=bg_fill)
            
            # 绘制文本（黑色）
            draw.text((left + 2, label_y), label_with_conf, fill=(0, 0, 0), font=font)
        
        # 转换回OpenCV格式（BGR）
        frame_rgb = np.array(pil_image)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        return frame_bgr

    def _draw_all_faces_labels(self, frame, face_locations: List, detected_students: List[Dict]):
        """
        在图像上绘制“所有检测到的人脸框”，并对已识别学生显示简化标签：
        - 已识别：姓名 + 学号前2位+末2位（例如 蒋建达2519）
        - 未识别：未知
        """
        if frame is None:
            return frame
        
        # 将OpenCV的BGR格式转换为RGB格式（PIL使用RGB）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # 尝试加载中文字体
        font_size = 20
        try:
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",
                "C:/Windows/Fonts/simsun.ttc",
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simkai.ttf",
            ]
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except Exception:
                        continue
            if font is None:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        
        # 构建“位置->识别信息”映射（用 location 完全一致来匹配）
        by_loc = {}
        for s in detected_students or []:
            loc = tuple(s.get('location', ()))
            if len(loc) == 4:
                by_loc[loc] = s

        def _status_for_student_id(student_id: str) -> str:
            """返回 'confirmed' | 'suspected' | 'unknown'，基于全程统计。"""
            if not student_id:
                return "unknown"
            obs = self.student_observations.get(str(student_id))
            best_conf = float(obs.get('best_confidence', 0.0)) if isinstance(obs, dict) else 0.0
            confirmed = (str(student_id) in self.detected_students) and (best_conf >= float(self.min_confirm_best_confidence))
            return "confirmed" if confirmed else "suspected"

        # 先用 OpenCV 画所有人脸框（颜色按“全程总结”着色）
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        for loc in face_locations:
            top, right, bottom, left = loc
            s = by_loc.get(tuple(loc))
            if s:
                status = _status_for_student_id(str(s.get('student_id', '')))
                if status == "confirmed":
                    color = (0, 255, 0)      # BGR 绿色：确认
                else:
                    color = (255, 0, 0)      # BGR 蓝色：疑似
            else:
                color = (0, 255, 255)        # BGR 黄色：未知
            cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)
        
        # 回到 PIL 画文本
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        for loc in face_locations:
            top, right, bottom, left = loc
            s = by_loc.get(tuple(loc))
            if s:
                name = s.get('name', '未知')
                student_id = s.get('student_id', '')
                student_id_clean = str(student_id).rstrip('.0')
                digits_only = ''.join([c for c in student_id_clean if c.isdigit()])
                short_id = student_id_clean
                if len(digits_only) >= 4:
                    short_id = f"{digits_only[:2]}{digits_only[-2:]}"
                confidence = float(s.get('confidence', 0))
                label = f"{name}{short_id}"
                
                status = _status_for_student_id(str(student_id))
                if status == "confirmed":
                    label_text = f"{label} {confidence:.2f}"
                    bg_color = (0, 255, 0)  # 绿色背景：确认
                else:
                    # 未确认的匹配在汇总图中降级为“疑似”，避免把误识别当成最终结果
                    label_text = f"疑似:{label} {confidence:.2f}"
                    bg_color = (0, 0, 255)  # 蓝色背景：疑似（未达确认门槛）
            else:
                label_text = "未知"
                bg_color = (255, 255, 0)  # 黄色背景：未识别
            
            # 计算文本大小
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 放在框上方/下方
            label_y = top - text_height - 5 if top - text_height - 5 > 0 else bottom + 5
            bg_left = left
            bg_top = label_y - 2
            bg_right = left + text_width + 4
            bg_bottom = label_y + text_height + 2
            draw.rectangle([bg_left, bg_top, bg_right, bg_bottom], fill=bg_color)
            draw.text((left + 2, label_y), label_text, fill=(0, 0, 0), font=font)
        
        frame_rgb = np.array(pil_image)
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def _create_summary_board_image(self):
        """
        生成“总结面板图”（纯文字），展示全程统计后的确认/疑似名单。
        - 绿色：确认出勤（hits≥min_presence_hits 且 best_confidence≥min_confirm_best_confidence）
        - 蓝色：疑似（出现过但未达到确认门槛）
        """
        # 没有任何观测就不生成
        if not self.student_observations:
            return None

        # 构建名单
        confirmed_rows = []
        suspected_rows = []
        for student_id, obs in self.student_observations.items():
            hits = int(obs.get('hits', 0))
            best_conf = float(obs.get('best_confidence', 0.0))
            confirmed = (student_id in self.detected_students) and (best_conf >= float(self.min_confirm_best_confidence))

            # 学号简化：前2位+末2位
            student_id_clean = str(student_id).rstrip('.0')
            digits_only = ''.join([c for c in student_id_clean if c.isdigit()])
            short_id = student_id_clean
            if len(digits_only) >= 4:
                short_id = f"{digits_only[:2]}{digits_only[-2:]}"

            row = f"{short_id}  hits={hits}  best={best_conf:.2f}"
            if confirmed:
                confirmed_rows.append(row)
            else:
                suspected_rows.append(row)

        # 排序：先按 best_conf 再按 hits
        def _sort_key(line: str):
            # line like: "2519  hits=3  best=0.42"
            try:
                parts = line.split()
                hits = int(parts[1].split("=")[1])
                best = float(parts[2].split("=")[1])
                return (-best, -hits, parts[0])
            except Exception:
                return (0, 0, line)

        confirmed_rows.sort(key=_sort_key)
        suspected_rows.sort(key=_sort_key)

        # 画布大小：自适应行数，但限制最大高度，避免太长
        width = 1100
        header_h = 90
        line_h = 26
        max_lines = 40  # 每栏最多显示行数，更多可通过提高 max_frames/导出CSV查看
        lines_left = min(max_lines, len(confirmed_rows))
        lines_right = min(max_lines, len(suspected_rows))
        body_lines = max(lines_left, lines_right)
        height = header_h + body_lines * line_h + 40

        # PIL 画图
        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        # 字体
        font_size = 20
        try:
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simhei.ttf",
                "C:/Windows/Fonts/simsun.ttc",
            ]
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except Exception:
                        continue
            if font is None:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # 标题
        title = "出勤总结（全程统计）"
        params = f"确认: hits≥{self.min_presence_hits}, 单帧计入hits: conf≥{self.hit_confidence_threshold:.2f}, best_conf≥{self.min_confirm_best_confidence:.2f}"
        draw.text((20, 15), title, fill=(0, 0, 0), font=font)
        draw.text((20, 45), params, fill=(80, 80, 80), font=font)

        # 分栏标题
        draw.rectangle([20, 70, width // 2 - 10, 70 + 30], fill=(0, 255, 0))
        draw.rectangle([width // 2 + 10, 70, width - 20, 70 + 30], fill=(0, 0, 255))
        draw.text((25, 74), f"确认出勤（最多显示 {max_lines} 条）: {len(confirmed_rows)}", fill=(0, 0, 0), font=font)
        draw.text((width // 2 + 15, 74), f"疑似（最多显示 {max_lines} 条）: {len(suspected_rows)}", fill=(255, 255, 255), font=font)

        # 内容
        y0 = 105
        for i in range(body_lines):
            if i < lines_left:
                draw.text((25, y0 + i * line_h), confirmed_rows[i], fill=(0, 120, 0), font=font)
            if i < lines_right:
                draw.text((width // 2 + 15, y0 + i * line_h), suspected_rows[i], fill=(0, 0, 180), font=font)

        # 转 BGR
        arr = np.array(img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def _create_final_summary_image(self, frame_bgr, face_locations: List, detected_students: List[Dict]):
        """
        生成“最终总结图”：上半部分为总结面板，下半部分为参考帧（按全程总结着色）。
        """
        board = self._create_summary_board_image()
        annotated = None
        if frame_bgr is not None:
            annotated = self._draw_all_faces_labels(frame_bgr.copy(), face_locations or [], detected_students or [])
        if board is None and annotated is None:
            return None
        if board is None:
            return annotated
        if annotated is None:
            return board

        # 统一宽度（以 board 为准），保持比例缩放
        bw = board.shape[1]
        ah, aw = annotated.shape[:2]
        if aw != bw:
            new_h = max(1, int(ah * (bw / float(aw))))
            annotated = cv2.resize(annotated, (bw, new_h), interpolation=cv2.INTER_AREA)

        pad = 12
        # 拼接：board + padding + annotated
        pad_img = np.full((pad, bw, 3), 255, dtype=np.uint8)
        return np.vstack([board, pad_img, annotated])
    
    def get_absent_students(self, all_student_ids: Set[str]) -> List[str]:
        """
        获取未到课学生名单
        
        Args:
            all_student_ids: 所有选课学生的ID集合
            
        Returns:
            未到课学生ID列表
        """
        absent_students = all_student_ids - self.detected_students
        return sorted(list(absent_students))
    
    def _get_student_name(self, student_list: Dict, student_id: str) -> str:
        """
        从学生列表中获取学生姓名（兼容新旧格式）
        
        Args:
            student_list: 学生信息字典
            student_id: 学生ID
            
        Returns:
            学生姓名
        """
        if student_id not in student_list:
            return "未知"
        
        student_info = student_list[student_id]
        if isinstance(student_info, dict):
            return student_info.get('name', '未知')
        else:
            # 旧格式：直接是姓名
            return student_info
    
    def generate_attendance_report(
        self, 
        student_list: Dict,
        output_path: str = "attendance_report.csv"
    ) -> pd.DataFrame:
        """
        生成出勤报告
        
        Args:
            student_list: 学生信息字典 {student_id: {'name': name}} 或 {student_id: name}
            output_path: 输出文件路径
            
        Returns:
            出勤报告DataFrame
        """
        all_student_ids = set(student_list.keys())
        absent_students = self.get_absent_students(all_student_ids)
        present_students = list(self.detected_students)
        
        # 创建报告数据
        report_data = []
        
        # 添加出勤学生
        for student_id in present_students:
            if student_id in student_list:
                report_data.append({
                    '学号': student_id,
                    '姓名': self._get_student_name(student_list, student_id),
                    '出勤状态': '出勤'
                })
        
        # 添加缺勤学生
        for student_id in absent_students:
            if student_id in student_list:
                report_data.append({
                    '学号': student_id,
                    '姓名': self._get_student_name(student_list, student_id),
                    '出勤状态': '缺勤'
                })
        
        # 创建DataFrame
        df = pd.DataFrame(report_data)
        df = df.sort_values('学号')
        
        # 保存到CSV
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"出勤报告已保存到: {output_path}")
        
        # 打印统计信息
        total_students = len(student_list)
        present_count = len(present_students)
        absent_count = len(absent_students)
        attendance_rate = (present_count / total_students * 100) if total_students > 0 else 0
        
        print(f"\n出勤统计:")
        print(f"总学生数: {total_students}")
        print(f"出勤人数: {present_count}")
        print(f"缺勤人数: {absent_count}")
        print(f"出勤率: {attendance_rate:.2f}%")
        
        return df

