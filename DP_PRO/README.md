## 课堂行为分析系统（任务1/任务2）

本项目当前**只维护任务1/任务2**的运行与结果输出：

- **任务1（出勤+座位图）**：用 InsightFace 做人脸识别，输出缺勤名单与 `attendance_images/seat_map.json`（供任务2约束定位）
- **任务2（课堂行为）**：按流程图重做版
  - `person`：框选目标 ->（CSRT/KCF）跟踪/重定位 -> YOLO Pose + 规则
  - `class`：每 N 秒采样 -> YOLO Pose +（可选）RT-DETR 物体 -> 规则 + 统计

### 环境要求

- Python 3.8–3.11（Windows 推荐 3.9）
- 建议使用 **PowerShell** 执行命令

### 安装依赖（推荐）

在 `C:\DP_PRO` 下：

```bash
python -m pip install -r requirements.txt
```

也可以直接运行脚本（已按当前依赖更新）：

```bash
install_dependencies.bat
```

### 数据准备（任务1需要）

- **选课名单**：`student_list.csv`（至少两列：`学号,姓名`）
- **学生照片（推荐预处理后）**：`student_photos_processed/`（文件名为 `学号.jpg`）
- **视频**：`教室视频/*.mp4`

照片预处理（会输出 `student_photos_processed/`，并处理中文路径）：

```bash
python preprocess_photos_insightface.py --input "学生照片目录" --output student_photos_processed --student-list student_list.csv
```

### 任务1：出勤检测 + 生成 seat_map

```bash
python -u main.py --video "教室视频/1105.mp4" --photos student_photos_processed --list student_list.csv --task 1 --start-time 63 --sample-interval 500 --max-frames 40
```

`--start-time` 支持：
- 分钟：`68` / `68.5`
- 时间戳：`HH:MM:SS`（例如 `01:08:00`）或 `MM:SS`（例如 `08:30`）

关键输出：

- `attendance_images/seat_map.json`（任务2会用）
- `attendance_images/*.jpg`（可视化结果，若你后续清理输出会被删除）

### 任务2：课堂行为（重做版）

#### person（个人模式，推荐）

弹窗框选 ROI（需要 `opencv-python`，不能是 headless）：

```bash
python -u main.py --video "教室视频/1105.mp4" --task 2 --t2-mode person --t2-start-minute 68 --t2-duration 10 --t2-sample-seconds 10 --t2-interactive-roi --t2-save-marked-images --t2-device cuda
```

`--t2-start-minute` 同样支持分钟/时间戳（`68` 或 `01:08:00`）。

如果不想弹窗，可手动指定 ROI：

```bash
python -u main.py --video "教室视频/1105.mp4" --task 2 --t2-mode person --t2-roi "468,388,130,172" --t2-start-minute 68 --t2-duration 10 --t2-sample-seconds 10 --t2-save-marked-images --t2-device cuda
```

#### class（全班模式）

```bash
python -u main.py --video "教室视频/1105.mp4" --task 2 --t2-mode class --t2-start-minute 68 --t2-duration 10 --t2-sample-seconds 10 --t2-save-marked-images --t2-device cuda
```

任务2输出（固定到 `task2_outputs/`）：

- `task2_outputs/task2_results.json`
- `task2_outputs/images/`（加 `--t2-save-marked-images` 才会保存）

### 打包成带 Qt 前端的 EXE（CPU 离线版）
目标：双击 `DP_PRO_GUI.exe`，在窗口里选择视频/参数，运行任务1和任务2，并在日志面板看到进度。

#### 1) 准备离线模型（首次可能需要联网）
在项目目录执行：

```bash
python prepare_offline_assets.py
```

会生成：
- `assets/weights/yolov8s-pose.pt`
- `assets/weights/rtdetr-l.pt`
- `assets/insightface/models/...`（buffalo_l）

#### 2) 一键打包
在项目目录执行：

```bash
build_gui_exe.bat
```

输出：
- `dist/DP_PRO_GUI/DP_PRO_GUI.exe`

#### 3) 离线机制说明
- **任务2**：若 `assets/weights/` 存在，GUI/CLI 会优先使用离线 `.pt`，不会联网下载。
- **任务1**：若 `assets/insightface/models/` 存在，会自动设置 `INSIGHTFACE_HOME=assets/insightface` 使用离线 InsightFace 模型。

### GPU 说明（Windows）

- **任务2（Ultralytics）**：要用 GPU 需要安装 CUDA 版 PyTorch，命令里使用 `--t2-device cuda` 或 `--t2-device 0`
- **任务1（InsightFace）**：可选安装 `onnxruntime-gpu` 加速（不影响正确性）

### 常见问题

- ROI 弹窗报 `cvNamedWindow` / `The function is not implemented`：你装了 `opencv-python-headless`，卸载并安装 `opencv-python`
- 依赖冲突（numpy/protobuf/onnx）：请按 `requirements.txt` 固定版本安装
- `.pt` 权重下载损坏：删除同名 `.pt` 后重试让 Ultralytics 重新下载

### 清理输出

可安全删除（会重新生成/重新下载）：

- `task2_outputs/`
- `attendance_images/*.jpg`、`attendance_images/detections_cache.jsonl`
- 项目根目录下的 `*.pt`（下次任务2会重新下载）

不建议删除：

- `attendance_images/seat_map.json`（任务2 class / 基于座位约束需要）

