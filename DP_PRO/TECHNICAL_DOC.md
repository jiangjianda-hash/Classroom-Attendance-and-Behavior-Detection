## DP_PRO 技术文档（约9000字）

本文档面向需要维护/二次开发/部署本项目的工程人员，系统性说明 DP_PRO 的整体架构、任务1（出勤识别）与任务2（课堂行为分析）的算法流程与工程实现、离线模型与 EXE 打包方案，以及常见问题与排错建议。

> 约定：本文使用“时间点”表示视频中的某一位置，可用分钟（如 `68`）或时间戳（如 `01:08:00`）。项目已统一支持这两种输入（见 `main.py::parse_timepoint_to_minutes()`）。

---

### 1. 项目目标与边界

本项目聚焦两类业务输出：

- **任务1（出勤+座位图）**：对课堂视频在指定时间点附近抽帧，使用人脸识别识别学生身份，给出：
  - 出勤人数、出勤率
  - 缺勤人员名单（基于 `student_list.csv`）
  - `attendance_images/seat_map.json`（座位框与学号绑定），供任务2进行定位约束/初始化 ROI
- **任务2（课堂行为）**：对指定时间点后的固定时段（默认 10 分钟）进行行为统计：
  - `person`：针对单人 ROI（框选或 seat_map 初始化）做跟踪/重定位 + 姿态/规则推断
  - `class`：对全班每 N 秒采样一帧，做姿态识别与（可选）物体检测，输出全局/分人统计

边界与假设：

- 业务输入默认来自**固定教室机位**（相机角度与座位布局相对稳定）。
- 任务1需要**参照照片**（建议预处理并以 `学号.jpg` 命名）与**选课名单**（`student_list.csv`）。
- 任务2不强依赖任务1（可以不带名单/照片直接跑），但若存在 `seat_map.json` 能显著提升 person 模式的初始化体验与 class 模式的“按座位归因”质量。

---

### 2. 目录结构与关键文件

核心代码：

- `main.py`：CLI 主入口，包含：
  - `load_student_list()`：加载选课名单
  - `parse_timepoint_to_minutes()`：时间点解析（分钟/HH:MM:SS）
  - `task1_attendance_check()`：任务1流程封装（供 GUI/CLI 共用）
  - 任务2参数解析并调用 `task2_behavior_flow.run_task2()`
- `face_recognition_module.py`：InsightFace 封装、人脸库加载、provider 选择（GPU/CPU 兜底）
- `attendance_checker.py`：任务1视频抽帧、检测/识别、seat_map 生成、可视化标注与 CSV 报告
- `preprocess_photos_insightface.py`：参照照片预处理（中文路径读写、对齐、重命名为学号）
- `task2_behavior_flow.py`：任务2主流程（person/class 两套链路 + 输出）
- `behavior_rules_yolo.py`：行为规则引擎（由 pose 与 obj 结果推断 head_down/forward 与 phone/laptop/book 等）

GUI 与打包：

- `gui_app.py`：Qt（PySide6）窗口，选择视频与时间点后可一键跑任务1/2，并展示日志
- `prepare_offline_assets.py`：准备离线模型，输出到 `assets/`
- `dp_pro_gui.spec`：PyInstaller 规格文件（收集依赖模块与资源）
- `build_gui_exe.bat`：一键安装依赖、准备离线资源、打包 EXE
- `assets/`：离线资源目录（权重与 InsightFace 模型）

输出目录：

- `attendance_images/`：任务1输出（seat_map、标注帧、检测缓存）
- `task2_outputs/`：任务2输出（结果 JSON、汇总 JSON、标注图）

---

### 3. 数据准备与输入规范

#### 3.1 选课名单 `student_list.csv`

至少两列：**学号**、**姓名**。`main.py::load_student_list()` 会尽量识别列名（包含“学号/姓名”或 `student_id/name`），识别失败则使用前两列作为学号/姓名。

工程建议：

- 学号字段统一为字符串（避免 Excel 导出丢失前导 0）。
- CSV 编码使用 `utf-8-sig` 最稳（Windows/Excel 兼容）。

#### 3.2 参照照片 `student_photos_processed/`

强烈建议使用预处理脚本：

- 输入：原始照片目录（可含中文路径）
- 输出：`student_photos_processed/学号.jpg`

该约束可以让任务1更稳定地建立“学号 → 特征向量”的映射，避免文件名混乱导致错配。

#### 3.3 视频输入

- 支持 `mp4/avi/mov/mkv` 等常见格式。
- 对于中文路径，项目已在多个关键点采用 UTF-8 输出与 “imdecode/imencode + Python 文件 IO” 的方案以提升 Windows 兼容性。

---

### 4. 任务1：出勤识别与 seat_map 生成（算法与实现）

任务1的目标不是“全视频逐帧识别”，而是“在指定时间点附近抽帧，稳定识别尽可能多的学生，并构建 seat_map 供任务2使用”。

#### 4.1 流程总览

1) **加载名单**：`load_student_list(csv)` → `Dict[student_id] -> {name}`
2) **加载参照照片并建库**：`FaceRecognitionModule.load_student_photos(photos_dir, student_list)`
3) **视频抽帧**：在 `start_time_minutes` 处开始，每隔 `sample_interval` 帧采样一次，总采样 `max_frames` 次
4) **人脸检测 + 特征提取**：InsightFace `FaceAnalysis` 输出每张脸的 embedding
5) **匹配与累计 evidence**：
   - 与参照库做相似度/距离匹配
   - 维护每个学生的 hits（击中次数）、best_conf（最佳置信）
6) **seat_map 生成**：
   - 将识别结果与人脸框/座位区域关联（以供后续任务2按座位归因）
   - 冲突策略：同一座位出现多个学生时，保留更强 evidence（弱者被丢弃；不在输出中显式列出）
7) **输出**：
   - 标注帧与座位图（`attendance_images/*.jpg`）
   - `attendance_images/seat_map.json`
   - `attendance_report.csv`（出勤表）

#### 4.2 InsightFace 与 provider 选择

`face_recognition_module.py` 封装了 InsightFace 初始化与 providers 选择：

- 默认优先 GPU（CUDAExecutionProvider / DirectML），不可用则回退 CPU
- Windows 上常见的 `LoadLibrary failed with error 126` 多由缺失 CUDA/cuDNN DLL 或 VC++ runtime 引起
- 代码实现中包含：
  - **固定路径**追加（仅常见 CUDA 安装路径，不使用用户输入，避免安全风险）
  - 把 `torch\lib` 加入 DLL 搜索路径（很多机器上 cuDNN 由 PyTorch 自带）
  - **可加载性预检查**：若 `onnxruntime_providers_cuda.dll` 无法加载则不启用 CUDA provider，避免刷屏错误

CPU-only 交付（GUI EXE）场景中，建议设置 `INSIGHTFACE_FORCE_CPU=1`，完全避免 GPU provider 分支。

#### 4.3 中文路径读写（Windows）

OpenCV 在 Windows 下对包含中文的文件路径可能存在兼容问题。项目在参照照片预处理与部分输出环节采用：

- 读：`np.fromfile(path, dtype=np.uint8)` + `cv2.imdecode()`
- 写：`cv2.imencode()` + Python `open(path, "wb")`

这样可绕过 `cv2.imread/cv2.imwrite` 对路径编码的限制，稳定支持中文路径与中文文件名。

#### 4.4 关键参数与调参思路

任务1常用参数：

- `--start-time`：时间点（分钟/时间戳），建议选“学生基本坐定且露脸多”的时刻
- `--sample-interval`：采样间隔（帧），越小采样越密，但越慢
- `--max-frames`：采样次数，越大覆盖越多姿态/遮挡变化，有利于提升召回
- `--det-size`：InsightFace 检测输入尺寸，越大越容易检测远处小脸，但更慢
- `--tolerance`：匹配阈值，越严格误识别更少但可能漏人

工程建议（经验）：

- 想提升“人数识别全”：优先调大 `det-size`（如 960）与 `max-frames`，并适当减小 `sample-interval`
- 想减少“误认”：适当提高 `tolerance`（更严格）并增加多帧一致性（本项目 hits 机制即为此）

---

### 5. 任务2：课堂行为分析（person/class 两套链路）

任务2核心思想：利用 **人体姿态关键点（pose）** 推断“抬头/低头”，并在“低头”时结合 **桌面区域物体检测（obj）** 推断低头行为类别（手机/电脑/看书记笔记）。

#### 5.1 person 模式（单人）

输入：

- 视频 + 时间点
- ROI（手动输入 `x,y,w,h` 或 interactive ROI；也可使用 `seat_map.json + student_id` 初始化）

流程：

1) 取时间点附近的首帧，确定 ROI
2) 跟踪：CSRT/KCF 跟踪 ROI 内目标；必要时做重定位以应对漂移
3) pose：对目标区域进行姿态推理，拿到关键点（鼻子、肩膀、髋部等）
4) 行为规则：`behavior_rules_yolo.infer_behaviors()` 输出：
   - `head_down` / `looking_forward`
   - `objects.phone/laptop/book`（仅当 head_down=True 才允许为 True）
   - `behaviors.playing_phone/using_laptop/note_taking`（同上）
5) 统计：按采样点累计时长比例，输出中文汇总（抬头/低头/低头在做什么）

工程关键点：

- **“forward = 不是低头”**：已按需求调整，避免 forward 判定过严
- **低头时物体检测更准**：引入桌面 ROI 的高分辨率二次检测（`--t2-obj-roi-imgsz`），解决斜拍时电脑/书本小、全图检测不稳定的问题

#### 5.2 class 模式（全班）

输入：

- 视频 + 时间点（分析起点）
- 时长（默认 10 分钟）
- 采样间隔（秒）

流程：

1) 按 `sample_seconds` 采样一帧（例如每 10 秒）
2) pose：对整帧或 ROI 分块推理，得到多人框与关键点
3)（可选）obj：在 head_down 时做桌面 ROI 二次检测（与 person 同逻辑），并限制每帧最多处理 N 个低头的人（`--t2-obj-roi-max-people`）防止过慢
4) 规则引擎：对每个检测到的人输出 head_down/forward + 低头类别
5) 归因：
   - 若提供 `attendance_images/seat_map.json`，可用座位框与检测框 IoU 做贪心匹配，把行为归到某个 student_id
6) 汇总输出：
   - `task2_results.json`：逐样本记录
   - `task2_summary.json`：全局/按分钟/按学生统计，包含 `overall_cn`（听课比例、低头比例、低头类别比例）

#### 5.3 行为规则引擎（核心约束）

规则位于 `behavior_rules_yolo.py`，整体思路是“简单、可解释、可调参”：

- head_down：基于关键点几何关系/置信度与 EMA 平滑
- looking_forward：按需求定义为 `not head_down`
- phone/laptop/book：
  - 通过 obj 检测框与人体框（或桌面 ROI）进行归因
  - 设定最小置信度阈值，并对 phone 做更严格阈值与尺寸约束（降低误报）
  - **硬门控**：若 `head_down=False`，则 phone/laptop/book 必须为 False（避免“抬头也检测到手机”的噪声）

该设计的优点：

- 不依赖复杂训练数据即可在固定教室场景落地
- 输出可解释，便于现场调参
- 对部署环境要求相对低（CPU 也能跑，只是速度更慢）

---

### 6. 离线模型与资源管理

为了实现“离线可用 + 便于打包分发”，项目引入 `assets/` 资源目录：

推荐结构：

- `assets/weights/yolov8s-pose.pt`
- `assets/weights/rtdetr-l.pt`
- `assets/insightface/models/buffalo_l/...`

准备脚本：`prepare_offline_assets.py`

- 会触发 Ultralytics 下载 .pt 并拷贝到 `assets/weights/`
- 会初始化 InsightFace `FaceAnalysis(name="buffalo_l")` 触发下载，并拷贝 `~/.insightface/models` 到 `assets/insightface/models/`

运行时优先级：

- 任务2：若 `assets/weights` 存在且用户未显式修改模型路径，优先使用离线 `.pt`
- 任务1：若 `assets/insightface/models` 存在，自动设置 `INSIGHTFACE_HOME=assets/insightface`

注意：

- 离线资源准备阶段可能需要联网（首次下载）。
- 资产目录会显著增大体积（你已接受 2GB 上限，适合该方案）。

---

### 7. Qt GUI（PySide6）设计与线程模型

GUI 位于 `gui_app.py`，目标是把复杂的命令行参数收敛为“选视频 + 选时间点 + 选择模式 + 点击运行”。

关键设计点：

- 使用 `QThread`（`Worker`）在后台跑任务，避免阻塞 UI
- 将运行日志实时输出到窗口（通过 `_QtLogStream` 捕获 `print` 输出并发到文本框）
- 任务1完成后直接在弹窗/日志中输出：
  - 出勤人数/总人数/出勤率
  - 缺勤人数
  - 输出文件位置提示

当前默认选择 CPU-only：

- GUI 中默认设置 `INSIGHTFACE_FORCE_CPU=1`
- 任务2 device 固定为 `cpu`（满足“只要 CPU 也行”的交付要求）

后续可扩展：

- 增加“启用 GPU”开关（并提示依赖安装与环境要求）
- 增加 `--t2-interactive-roi` 的 ROI 弹窗开关（需确认 GUI OpenCV 环境）

---

### 8. EXE 打包（PyInstaller）

本项目提供一键脚本 `build_gui_exe.bat` 与 spec 文件 `dp_pro_gui.spec`。

#### 8.1 打包内容

- 主程序：`gui_app.py`
- 资源：`assets/`、`README.md`
- hiddenimports：收集 ultralytics/torch/onnxruntime/insightface/cv2 等子模块，降低运行时缺模块概率

输出位置：

- `dist/DP_PRO_GUI/DP_PRO_GUI.exe`

#### 8.2 典型问题

1) **首包很大**：深度学习依赖（torch/onnxruntime/opencv）+ 模型权重导致体积巨大，这是预期行为。
2) **运行时报缺 DLL**：CPU-only 模式通常不会触发 CUDA DLL，但仍可能受 VC++ runtime 影响；建议目标机器安装常规 VC++ 运行库。
3) **OpenCV GUI/Headless 冲突**：若用户需要 interactive ROI，必须确保使用 `opencv-python` 而不是 headless；项目已对该错误提供友好提示。

---

### 9. 性能与准确性建议（实战调参）

#### 9.1 任务1（出勤）

优先提高召回（更多人被识别）：

- 提高 `det-size`（例如 960）
- 增大 `max-frames`（例如 80）
- 减小 `sample-interval`（例如 250）
- `start-time` 选择“集体抬头/露脸多”的时刻

控制误认：

- 提高 `tolerance`（更严格）
- 保持多帧一致性（采样次数适当增大）

#### 9.2 任务2（行为）

提高“电脑/书本”检出：

- 提高 `--t2-obj-roi-imgsz`（例如 1280/1536）
- 若 class 模式低头人数多导致 ROI 检测被限额，可提高 `--t2-obj-roi-max-people`
- 若 RT-DETR 在你的教室场景不稳定，可替换 obj 模型为更强的 YOLO 检测模型（例如 `--t2-obj-model yolov8l.pt`），并同样放入 `assets/weights/`

降低“抬头也识别出手机”的误报：

- 本项目已采用“head_down 门控 + phone 更严格阈值/尺寸约束”，通常不需要额外动作
- 若仍误报，可提高 `--t2-obj-min-conf` 或在规则中继续收紧 phone 尺寸/位置约束

---

### 10. 常见错误与排错手册（Windows）

#### 10.1 `cv2.error: The function is not implemented (cvNamedWindow)`

原因：

- 装了 `opencv-python-headless`，不包含 HighGUI。

解决：

- 卸载 headless，并安装 GUI 版本的 OpenCV（如 `opencv-python==4.8.1.78`）。
- 或不使用弹窗 ROI，改用 `--t2-roi "x,y,w,h"`。

#### 10.2 InsightFace 导入失败（`FaceAnalysis 不可用`）

原因：

- InsightFace 版本链可能需要 `albumentations/qudida/scikit-image/scipy/PyYAML` 等。

解决：

- 按 `requirements.txt` 固定版本安装依赖。
- 若你需要避免安装回 headless OpenCV，按项目提示用 `--no-deps` 安装指定包。

#### 10.3 GPU provider `LoadLibrary failed with error 126`

原因：

- 缺少 CUDA/cuDNN DLL 或未在 PATH 中，或 VC++ runtime 缺失。

解决（CPU-only 交付可忽略）：

- 使用项目内的 CUDA PATH 兜底逻辑，或手动把 CUDA `bin` 与 `torch\\lib` 加入 PATH。
- 也可设置 `INSIGHTFACE_FORCE_CPU=1` 完全避开该分支。

---

### 11. 安全与工程规范（简述）

本项目在工程实现中遵循“安全输入与最小权限”原则：

- 不使用用户输入拼接系统命令或危险路径进行执行
- 只对固定、已知的常见 CUDA 安装路径做 PATH 兜底（不使用外部输入）
- 不硬编码任何密钥/账号等敏感信息

如果未来要扩展为“多人分发的桌面产品”，建议补充：

- 统一的日志分级与脱敏策略（避免将名单等敏感信息写入公开日志）
- 统一的错误上报与版本信息（便于现场排查）

---

### 12. 二次开发建议（扩展点）

#### 12.1 行为类别扩展

当前行为类别聚焦低头三类：**玩手机 / 看电脑 / 看书记笔记**。扩展建议：

- 在 `behavior_rules_yolo.py` 中新增对象类别映射与规则
- 在 `task2_behavior_flow.py` 统计部分新增字段与中文汇总
- 保持“head_down 门控”避免类别泛滥导致误报增多

#### 12.2 更强的检测模型

可替换：

- pose 模型：`yolov8s-pose.pt` → 更大模型（更准更慢）
- obj 模型：`rtdetr-l.pt` → `yolov8l.pt` 等（根据教室场景选择）

配合离线资产：

- 把新的 `.pt` 放入 `assets/weights/`，并在 GUI 默认值/README 示例中更新。

#### 12.3 GUI 体验增强

建议新增：

- “一键运行任务1→生成 seat_map → 立即运行任务2”的联动按钮
- 输出目录的“打开文件夹”按钮
- 任务2显示 `overall_cn` 的摘要面板（抬头/低头比例饼图等）

---

### 13. 结语

DP_PRO 的核心优势是：在固定教室场景下，用可解释、可控的工程策略实现“出勤 + 行为统计”的落地闭环，并通过离线模型与 EXE 打包降低使用门槛。后续迭代建议围绕“更稳的检测、可视化更友好、部署更简单”三条主线持续优化。

---

### 14. 参考项目的结构化写法：从“数据→训练→推理→评估”映射到 DP_PRO

你提供的参考项目（章节“学生课堂行为检测”）是典型的“数据采集/预处理 → 标注 → 训练（迁移学习）→ 推理 → 评估”的闭环写法。DP_PRO 当前的工程目标是“尽快在固定教室场景落地”，因此：

- **任务1**使用 InsightFace 的通用人脸识别模型（`buffalo_l`）做身份识别，并不需要自训。
- **任务2**使用 Ultralytics 的通用姿态模型（YOLOv8 Pose）与通用目标检测模型（RT-DETR-L / 可替换 YOLO）做“检测层”，再用规则把检测结果“翻译”为课堂行为统计。

但为了便于论文/技术评审/后续产品化，我们同样可以按参考项目的结构来写 DP_PRO，并补充“可选训练路线”（当通用模型在你教室里不够准时，如何用少量数据把模型针对性调优）。

本章从参考项目的逻辑出发，给出 DP_PRO 的“工程闭环”版本。

---

### 15. 数据采集与预处理（Task1/Task2 共用）

#### 15.1 数据来源与采集策略

DP_PRO 面向的原始数据是课堂监控/录播视频。与参考项目相同，数据质量决定了后续识别上限，重点关注：

- **机位稳定性**：固定教室机位（角度/焦距固定）最利于 seat_map 稳定。
- **画面清晰度**：分辨率越高，小目标（远处脸、桌面电脑、书本）越容易被检测到。
- **光照与遮挡**：逆光、反光、遮挡（手/书本遮脸）会明显降低任务1召回；对任务2会影响关键点置信度与桌面物体检出。
- **时间点选择**：本项目强调“时间点驱动”，即用户输入一个时间点，系统围绕该点完成任务1或任务2的统计。选择“学生就座且露脸较多”的时间点，会显著提升任务1的 seat_map 完整性。

#### 15.2 选课名单与学号体系

任务1以 `student_list.csv` 为“真值名单”，输出缺勤名单与出勤率。工程上需要保证：

- 学号唯一、稳定、不会因 Excel 处理而变成科学计数法。
- 姓名字段尽量一致（用于报表与 UI 展示）。

#### 15.3 学生参照照片预处理（任务1）

参考项目通常会在训练前做“数据清洗/增强”；DP_PRO 的对应环节是“参照照片标准化”，核心目标是：

- 让每个学生在参照库中都能产生稳定的人脸特征向量；
- 让文件命名与学号对齐，避免后续错配；
- 解决 Windows 中文路径/中文文件名带来的 OpenCV 兼容问题。

本项目的 `preprocess_photos_insightface.py` 做了三件关键事：

1) **中文路径读写兼容**：使用 `imdecode/imencode + Python 文件 IO`，绕过 `cv2.imread/cv2.imwrite` 在 Windows 下的编码限制。  
2) **人脸对齐/裁剪**：通过 InsightFace 生成的关键点或对齐逻辑，将人脸裁剪到稳定区域，提高跨光照/跨姿态的特征稳定性。  
3) **按名单重命名**：若提供 `student_list.csv`，输出文件统一命名为 `学号.jpg`（或 `学号.png`），直接与任务1的加载逻辑对齐。

工程建议（经验）：

- 每个学生尽量保证 1 张清晰正脸；若有多张，建议保留最清晰的一张。
- 若学生戴口罩/侧脸较多，可在课前采集“无口罩正脸”的参照照片以提升识别上限。

#### 15.4 视频时间点解析与统一

参考项目往往用“帧索引/时间戳”定位样本。DP_PRO 把这个抽象成统一的“时间点”，并支持：

- 分钟：`68` / `68.5`
- 时间戳：`HH:MM:SS` 或 `MM:SS`

实现位于 `main.py::parse_timepoint_to_minutes()`，GUI/CLI 共用，避免两套输入口径导致的偏差。

---

### 16. 任务1（出勤识别）工程细节：从“检测→识别→统计→座位图”

#### 16.1 为什么任务1不做“全视频识别”

参考项目若做行为检测训练，会尽量覆盖更多帧；而出勤场景的核心指标是“名单是否到齐”，并不需要逐帧识别。全视频逐帧识别会带来：

- 计算量巨大（尤其 CPU-only 部署）。
- 误识别累积（帧越多，越容易被偶发误检污染统计）。
- 可解释性差（很难回答“为什么认定某人到了/没到”）。

因此 DP_PRO 的策略是：围绕时间点抽取有限数量的代表帧，在每一帧里识别尽可能多的学生，通过 hits 累计构建稳定证据，最终输出缺勤名单与 seat_map。

#### 16.2 人脸检测与特征提取（InsightFace buffalo_l）

InsightFace 的 `FaceAnalysis(name="buffalo_l")` 通常包含：

- 人脸检测（det_10g）
- 人脸识别特征（w600k_r50）
- 关键点/属性（genderage、landmark）

本项目通过 `face_recognition_module.py` 做了工程层增强：

- **det_size 规范化**：调整到 32 的倍数，避免部分模型内部尺寸不匹配。
- **小图参照照片兜底**：对 512×512 等小图使用更小 det_size 再试；必要时 2× 放大再检一次，提高参照图召回。
- **GPU/CPU provider 选择**：优先 CUDA/DirectML，失败则回退 CPU；对 Windows “error 126”提供 PATH 兜底。

#### 16.3 人脸匹配与阈值解释

InsightFace 输出的 embedding 是归一化向量，常用距离是：

- 余弦相似度：\(s = \langle e_1, e_2 \rangle\)
- 余弦距离：\(d = 1 - s\)（范围约 \([0, 2]\)，越小越像）

项目的 `tolerance` 在工程上等价于“距离阈值”：距离小于阈值则判为匹配。为什么用户会感觉 `tolerance` 越大越严格/越小越严格？这是历史遗留的命名差异：有些库把阈值定义为“相似度阈值”，但本项目内部按“距离阈值”理解并在日志中说明。实际使用以“误识别/漏识别”现象为准进行调参即可。

#### 16.4 hits / best_conf：把“偶发识别”变成“稳定证据”

参考项目在训练/评估里会用 mAP 等指标量化；任务1属于工程统计，核心是稳定性。本项目用两个关键统计字段：

- `hits`：该学生在采样帧里被识别为最佳匹配的次数
- `best_conf`：该学生所有匹配中最强的一次置信（可理解为最小距离/最大相似度的映射）

工程意义：

- **hits 高**：说明在多帧、多姿态下都能稳定匹配，属于“确认到场”的强证据。
- **hits 低但 best_conf 强**：可能是遮挡/侧脸导致多数帧无法匹配，但偶尔有一帧很清晰；属于“疑似到场”，需要在可视化上做区分（你之前要求的蓝色标注即源于此）。

#### 16.5 seat_map 的生成与冲突处理

`seat_map.json` 的用途是把“空间位置（座位区域）”与“身份（学号）”绑定。这样任务2可以：

- person：输入学号 → 自动定位 ROI
- class：把每个人的行为归到某个学号（座位约束）

冲突的本质是：同一座位区域内可能出现多个学生候选（误检/遮挡/走动）。DP_PRO 当前策略是：

- 在冲突时保留 evidence 更强的学生，弱证据丢弃（不显式输出 dropped 列表，保持输出简洁）。

在产品化上，冲突处理的可解释性可以通过“调试模式/可选输出”增强，但默认对用户隐藏复杂细节，以保证结果可读。

---

### 17. 任务2（课堂行为分析）工程细节：从“姿态+物体→规则→统计”

#### 17.1 行为定义：从参考项目的 6 类到 DP_PRO 的 3 类

参考项目举例的 6 类行为包括：举手、阅读、记笔记、玩手机、低头、趴桌等。DP_PRO 当前聚焦你明确提出的三类低头行为：

- **低头看手机**
- **低头看电脑**
- **低头看书/记笔记**

并以“是否低头”作为第一层状态：

- `head_down=True`：进入“低头类别”判别
- `head_down=False`：直接计入“抬头/向前看（forward）”

这样做的原因：

- 教室固定机位里，“举手/阅读/写字”等细粒度动作需要更强的关键点质量或专门训练的行为模型；
- 你当前的业务诉求集中在“低头时在做什么”，三类可用“桌面物体”强约束来实现可解释判别；
- 用规则+通用模型可以快速落地，并且便于按教室角度调参。

#### 17.2 person 模式：跟踪/重定位 + ROI 桌面二次检测

person 模式的挑战是：目标可能轻微移动、ROI 可能漂移。项目采取：

- CSRT/KCF 跟踪（可选）
- EMA 平滑（对 head_down 分数平滑，避免抖动）
- **桌面 ROI 二次检测**（高分辨率）：在低头时仅对桌面区域做一次更高 imgsz 的检测，提高电脑/书本检出率

你之前提到“低头时识别不到书/电脑”，本质通常是：

- 全图尺度下桌面目标太小；
- RT-DETR 对斜拍桌面笔记本的外观泛化不足；
- 桌面 ROI 归因门控过严（物体框被判为不属于该人）。

我们已经用“扩大桌面归因区域 + ROI 二次检测 + phone 更严格阈值/尺寸约束 + head_down 门控”把这条链路做得更稳定。

#### 17.3 class 模式：多人体姿态 + 低头才做 ROI 物体检测（限额）

class 模式要处理全班多人，直接对每帧跑全图 obj 很慢且误报多。DP_PRO 的工程取舍是：

- 姿态（pose）每帧必跑：决定谁在低头
- 物体（obj）只对低头的人跑桌面 ROI（二次检测更准），并设置每帧上限 `--t2-obj-roi-max-people` 防止极端场景变慢

同时输出 `overall_cn`：

- 抬头比例
- 低头比例
- 低头类别比例（手机/电脑/书/其他）

这与参考项目“对课堂学习状态进行整体评估”的思路一致，只是 DP_PRO 输出的是可直接用于业务报表的统计结果。

#### 17.4 规则引擎：为什么要“门控（gating）”

在真实教室视频里，目标检测会出现：

- 桌面上某些物品被误检为手机；
- 电脑屏幕反光/键盘形状被误判；
- 书本被遮挡时漏检；
- 人与物体的空间关系复杂（斜拍/遮挡/多人重叠）。

因此 DP_PRO 的规则引擎强调“强约束与可解释”：

- **head_down 门控**：只在低头时允许 phone/laptop/book 为 True（你明确提出的业务规则）
- **phone 更严格**：最小置信度阈值取 `max(obj_min_conf, 0.35)`，并加尺寸约束，减少把桌面杂物当手机
- **桌面归因区域扩展**：覆盖 45 度机位下更大的桌面范围，提升电脑/书本被归因到人的概率

这类门控相当于把参考项目里“通过数据与训练学习到的先验”用工程规则显式写出来，优点是无需额外标注数据即可快速提升稳定性。

---

### 18. 可选训练路线（当通用模型不够准时）

> 说明：DP_PRO 当前默认不需要训练即可使用。本章提供“可选路线”，用于你未来希望把模型更贴合某个教室/机位的场景。

参考项目采用 DAMO-YOLO 并在 ModelScope 做迁移学习。DP_PRO 的可选训练路线可以更贴近你现有依赖栈：

- 姿态：继续使用 YOLOv8 Pose（通常不需要自训，除非关键点在你机位下质量很差）
- 物体：可用 YOLOv8 检测模型（如 `yolov8l.pt`）进行微调，让“笔记本/书本/手机”更贴合你的教室桌面外观
- 行为：若未来要做“举手/阅读/写字/趴桌”等细粒度动作，建议训练一个轻量行为分类模型（输入为人的上半身或桌面 ROI），或训练一个多类别检测模型（直接检测行为类别）

#### 18.1 数据采样与截帧

从课堂视频中采样若干时间段，按固定间隔抽帧（例如每 1–2 秒一帧），保证：

- 包含不同光照、不同坐姿、不同遮挡
- 覆盖手机/电脑/书本在桌面上的多种摆放方式
- 覆盖不同座位距离（近处/远处）

#### 18.2 标注工具与格式

参考项目使用 LabelImg 标注并输出 XML；也提到 COCO/YOLO 格式转换。对 DP_PRO 的物体检测训练建议：

- 标注工具：LabelImg / Labelme 均可
- 建议格式：YOLO txt（训练 YOLOv8 最直接）
- 类别建议从小做起：`phone`, `laptop`, `book`（必要时可扩展 `tablet`, `notebook_paper`, `calculator` 等）

#### 18.3 训练与迁移学习（以 Ultralytics YOLO 为例）

在工程上，迁移学习的核心是：

- 用预训练权重初始化（例如 `yolov8l.pt`）
- 用你教室的数据做少量 epoch 微调
- 通过验证集观察 mAP 与误报/漏报的变化

对于本项目的目标，评估重点不是极致 mAP，而是：

- 在“低头桌面 ROI”里是否能稳定检出笔记本/书本/手机；
- 是否减少“抬头也被误判手机”的噪声（规则门控能缓解，但从模型层降低误报更好）。

#### 18.4 与 DP_PRO 的集成方式

训练完的新权重（例如 `best.pt`）集成非常简单：

- 把权重放入 `assets/weights/`
- 在 GUI/CLI 中把 `obj_model` 改为该权重
- 若仍需离线分发，重新运行 `build_gui_exe.bat` 打包即可

---

### 19. 输出文件格式说明（面向联调/二次开发）

本节是参考项目里“COCO 格式标签/推理结果展示”的对应部分：明确 DP_PRO 的关键输出结构，便于你做二次开发、报表、或与教务系统联动。

#### 19.1 `attendance_images/seat_map.json`（任务1）

用途：把座位区域与学号绑定，供任务2定位/归因。

字段（高层）：

- `video`: 源视频路径（或文件名）
- `timestamp`: seat_map 生成时对应的采样点
- `seats`: 座位列表，每个包含：
  - `seat_id`: 座位编号（工程内部生成）
  - `student_id`: 绑定学号（可能为空）
  - `box`: 座位区域框（x,y,w,h 或 x1,y1,x2,y2，具体以文件为准）
  - `confidence`: 绑定置信（若存在）

#### 19.2 `task2_outputs/task2_results.json`（任务2逐样本）

用途：逐帧（逐采样）记录，每条记录可用于回放、调试、或自定义统计。

常见字段：

- `mode`: `person` / `class`
- `frame_idx`: 帧号
- `time_sec` / `minute`: 时间信息
- `person_box`: 人框
- `kps`: 关键点
- `beh`: 行为结果：
  - `head_down`, `looking_forward`
  - `objects.phone/laptop/book`
  - `behaviors.playing_phone/using_laptop/note_taking`
  - `objects_debug`（调试字段，可选）

#### 19.3 `task2_outputs/task2_summary.json`（任务2汇总）

用途：直接给业务侧看的统计结果（中文字段），包括：

- person：抬头/低头比例，低头类别比例
- class：整体 `overall_cn`（听课比例、低头比例、低头类别比例），以及按分钟/按学生聚合

---

### 20. 部署与交付建议（从工程到产品）

参考项目通常停留在“训练+推理脚本”。DP_PRO 已经进一步产品化到“GUI + EXE”：

- 用户只需要：选择视频 + 输入时间点 + 点击运行
- 任务1输出：出勤人数/出勤率/缺勤名单 + seat_map
- 任务2输出：指定时间点后 10 分钟内的课堂行为统计（person/class）

为了让交付更稳，建议：

- **默认 CPU**（尤其任务1）：避免用户机器 CUDA/cuDNN 依赖不全导致失败
- **任务2 GPU 可选**：多数机器只要显卡驱动正常，torch CUDA 跑起来相对更容易
- **离线 assets 随包**：保证无网络也能跑（你已接受 2GB 体积）
- **统一输出目录**：便于教务系统/报表系统拿结果

---

### 21. 附录：对齐参考项目的“章节要点”到 DP_PRO

为了让你的文档/汇报更像参考项目那种章节写法，这里给一个对齐清单（你可以直接用于 PPT 目录）：

- 第 1 章：项目目标与应用价值（DP_PRO 任务1/2）
- 第 2 章：数据采集与预处理（视频采样、名单/照片标准化、中文路径）
- 第 3 章：算法与模型选择（InsightFace、YOLO Pose、RT-DETR/YOLO Obj、规则引擎）
- 第 4 章：系统实现（流程图、模块划分、关键代码路径）
- 第 5 章：实验与评估（出勤召回/误识别、行为统计一致性、速度/资源）
- 第 6 章：部署与交付（GUI/EXE、离线 assets、CPU/GPU 策略、排错）
- 第 7 章：总结与展望（扩展行为类别、可选训练路线、教室迁移）

---

> 注：本文已在原技术文档基础上补齐“参考项目式”的数据/训练/推理/评估写法，并明确了 DP_PRO 的现状（以工程落地为主）与可选训练路线（面向未来迭代）。如你需要“严格不少于 9000 字”的形式化统计，我也可以再补一个“字数统计与版本记录”小节，确保审稿/验收无争议。

---

### 22. 代码附录（关键实现清单，含原仓库代码片段）

本节按“可读性 + 可复用性 + 可排错性”的原则，摘录 DP_PRO 的关键代码片段，并解释每段代码的设计动机、输入输出、常见坑与可扩展点。

> 说明：以下代码均来自本仓库当前版本（Task1/Task2/GUI/打包相关），为了便于阅读，部分片段会省略与主题无关的 import 或支撑函数。

---

#### 22.1 时间点解析（统一输入口径）

文件：`main.py`  
用途：把 GUI/CLI 的“时间点”（分钟或 `HH:MM:SS`）统一转为分钟数（float），保证任务1/2的起点一致。

```python
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

    nums = [float(p) for p in parts]
    if len(nums) == 2:
        mm, ss = nums
        hh = 0.0
    else:
        hh, mm, ss = nums

    if hh < 0 or mm < 0 or ss < 0:
        raise ValueError(f"时间点不能为负数: {s}")
    if ss >= 60 or mm >= 60:
        raise ValueError(f"时间点分/秒需 < 60: {s}")

    return float(hh * 60.0 + mm + ss / 60.0)
```

设计要点：

- **拒绝歧义**：`MM:SS` 的 `MM` 必须 `<60`，避免用户输入 `90:00` 产生歧义（它到底是 90 分钟还是 90 秒？）。
- **安全约束**：仅做数值解析，不拼路径、不执行命令，符合仓库安全规范。

---

#### 22.2 任务1入口（GUI/CLI 共用）

文件：`main.py`  
用途：把任务1（出勤识别）封装成函数，GUI 直接调用（不在 GUI 内硬解析 argparse）。

```python
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

    face_module = FaceRecognitionModule(
        tolerance=float(tolerance),
        model_type="insightface",
        det_size=(int(det_size), int(det_size)),
    )
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
    print(f"处理帧数: {result.get('total_frames_processed')}")
    return report_df, absent
```

设计要点：

- **依赖延迟导入**：在函数内导入任务1依赖，便于在缺包时给用户更清晰的错误提示（且 GUI 不会一启动就因为 InsightFace 依赖链崩溃）。
- **输出固定目录**：避免把用户输入直接用于文件路径（安全策略），同时方便任务2复用 `seat_map.json`。

---

#### 22.3 任务1核心：视频抽帧 + 观测统计（hits/best_conf）+ seat_map 生成

文件：`attendance_checker.py`

（1）抽帧与缓存（节选）

```python
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(start_time_minutes * 60 * fps) if fps > 0 else 0
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# 可选：写“任务1检测缓存”，供任务2复用（避免重复做人脸识别）
if output_images_dir:
    cache_path = os.path.join(output_images_dir, "detections_cache.jsonl")
    meta_path = os.path.join(output_images_dir, "detections_meta.json")
    # 注意：缓存文件名固定，避免将外部输入直接用于文件路径（安全策略）。
```

（2）稳定 seat_map 的生成（节选）

```python
# 生成“稳定座位位置地图”（供任务2直接定位）
seat_map = {}

for sid, samples in (self.student_location_samples or {}).items():
    locs = [s["location"] for s in samples if s.get("location")]
    # 分桶聚类（按 80px 网格），选频次最高的簇
    bin_size = 80
    bins = {}
    for loc in locs:
        cx, cy = _center(loc)
        key = (int(cx) // bin_size, int(cy) // bin_size)
        bins[key] = bins.get(key, 0) + 1
    best_key = max(bins.items(), key=lambda kv: kv[1])[0]

    # 在簇附近保留样本，用中位数框作为最终 estimated_location
    kept = []
    kept_conf = []
    for s in samples:
        loc = s.get("location")
        if not loc:
            continue
        cx, cy = _center(loc)
        # 与簇中心距离门控
        if d2 <= max_dist2:
            kept.append(loc)
            kept_conf.append(float(s.get("confidence", 0.0)))

    est_loc = [int(med[0]), int(med[1]), int(med[2]), int(med[3])]  # t,r,b,l
    obs = self.student_observations.get(str(sid), {}) or {}
    hits = int(float(obs.get("hits", 0.0))) if obs else 0
    best_conf = float(obs.get("best_confidence", 0.0)) if obs else 0.0

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
```

（3）冲突消解：同一座位多学生候选（节选）

```python
# 位置冲突消解：避免同一位置出现多个学生标注
items.sort(key=lambda kv: _strength(kv[1]), reverse=True)
kept: List[tuple] = []
used_bins = set()
for sid, info in items:
    loc = info.get("estimated_location")
    cx, cy = _center(loc)
    key = (int(cx) // bin_size, int(cy) // bin_size)
    conflict = False
    if key in used_bins:
        conflict = True
    else:
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
```

工程解释：

- **为什么要聚类**：单帧的人脸框会抖动，直接取某一帧位置会导致 `seat_map` 不稳定；用“多帧聚类 + 中位数”能大幅提升稳定性。
- **为什么要冲突消解**：错误识别会把两个学生的候选位置聚到同一座位；冲突消解能避免 seat_map 输出“一处两个名字”，影响任务2归因。

---

#### 22.4 任务2配置与入口（person/class 两种链路）

文件：`task2_behavior_flow.py`

（1）配置数据结构（节选）

```python
@dataclass
class Task2Config:
    video_path: str
    mode: str  # "person" | "class"
    start_minute: float
    duration_minutes: float
    sample_seconds: float
    tracker: str
    roi: Optional[Tuple[int, int, int, int]]
    interactive_roi: bool
    output_dir: str
    output_json: str
    save_images: bool
    pose_model: str
    obj_model: str
    device: str  # auto/cuda/cpu
    imgsz: int
    obj_every: int
    no_obj: bool
    obj_roi_imgsz: int
    obj_roi_max_people: int
    obj_min_iou: float
    obj_min_conf: float
    seat_map_path: str
    student_id: Optional[str]
```

（2）OpenCV 弹窗 ROI（CLI 传统方式，节选）

```python
def _select_roi_interactive(frame_bgr: np.ndarray, max_disp_w: int = 1920, max_disp_h: int = 1080) -> Tuple[int, int, int, int]:
    title = "Select Target ROI"
    h0, w0 = frame_bgr.shape[:2]
    scale = min(max_disp_w / float(w0), max_disp_h / float(h0), 1.0)
    disp = cv2.resize(frame_bgr, ...) if scale < 1.0 else frame_bgr

    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    except cv2.error as e:
        # 常见于 opencv-python-headless（无 highgui）
        raise RuntimeError(
            "当前 OpenCV 不包含 GUI/HighGUI 支持，无法使用 --t2-interactive-roi 弹窗框选。\n"
            "解决方案：卸载 headless 并安装 opencv-python，或改用 --t2-roi。"
        ) from e

    roi = cv2.selectROI(title, disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    # 映射回原图坐标并 clamp
    return x, y, w, h
```

注意：我们已经在 GUI 中改成 **Qt 自己框选 ROI**（见 22.7），因此 EXE 不再依赖 HighGUI。

---

#### 22.5 行为规则引擎：head_down/forward + 低头三分类（手机/电脑/书）

文件：`behavior_rules_yolo.py`

（1）forward 的定义（按你的业务要求：不是低头就是 forward）

```python
# 你定义的 forward：只要“不是低头”就算 forward。
# 因此 forward 的分数直接取 (1 - score_down)
score_forward = float(_clamp01(1.0 - float(score_down)))
head_status = "down" if float(score_down) >= 0.60 else "forward"
head_down = bool(head_status == "down")
looking_forward = bool(head_status == "forward")
```

（2）低头门控（关键业务规则：抬头时强制 phone/laptop/book=False）

```python
# 用户要求：只有在“低头”时，phone/laptop/book 才允许为 True
if not bool(head_down):
    has_phone = False
    has_laptop = False
    has_book = False
```

（3）phone 更严格阈值与尺寸约束（降低误报）

```python
phone_conf_thr = max(float(obj_min_conf), 0.35)

def _phone_size_ok(b):
    return (_area_xyxy(b) / region_area) <= 0.18

has_phone = any(
    o.cls_name in {"cell phone", "mobile phone", "phone"}
    and float(o.conf) >= phone_conf_thr
    and _phone_size_ok(o.xyxy)
    for o in rel_objs
)
```

工程解释：

- 你的场景里“降低 obj_min_conf 以提升书本召回”会引入大量 phone 误检；因此 phone 单独设更严格下限与尺寸门槛。
- `objects_debug` 字段用于现场定位问题：到底是“模型没检出 laptop”还是“检出但被门控过滤/归因失败”。

---

#### 22.6 任务2设备选择（CPU/GPU）与自动纠正（UltralyticsBackend）

文件：`task2_behavior_flow.py`（节选）

```python
try:
    import torch
    d = (device or "").strip().lower()
    if d in {"", "auto"}:
        resolved_device = "0" if bool(torch.cuda.is_available()) else "cpu"
    elif d in {"cuda", "gpu"}:
        resolved_device = "0" if bool(torch.cuda.is_available()) else "cpu"
    elif d == "cpu":
        resolved_device = "cpu"
    else:
        if not bool(torch.cuda.is_available()) and d != "cpu":
            resolved_device = "cpu"
except Exception:
    resolved_device = "cpu"
```

设计要点：

- **“用户想用 cuda 但机器无 CUDA”**：自动回退 CPU，避免直接崩溃。
- **GUI/EXE 交付**：默认 CPU 最稳；任务2启用 GPU 更常见更容易（torch CUDA）。

---

#### 22.7 GUI：Qt ROI 框选（不依赖 OpenCV HighGUI，EXE 更稳定）

文件：`gui_app.py`

（1）在时间点读一帧（用于 ROI 框选底图）

```python
def _read_frame_at_timepoint(video_path: str, timepoint: str) -> np.ndarray:
    minute = float(parse_timepoint_to_minutes(timepoint))
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_idx = int(max(0.0, minute * 60.0 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"无法读取视频帧（timepoint={timepoint}, frame={frame_idx}）")
    return frame
```

（2）Qt 对话框 ROI 框选（核心交互）

```python
class RoiSelectDialog(QtWidgets.QDialog):
    """
    Qt ROI 框选对话框：显示一帧图像，鼠标拖拽得到 ROI（原图坐标系）。
    这样不依赖 OpenCV HighGUI，打包后更稳定。
    """
    # ... 省略：显示 pixmap、rubber band、坐标映射 ...
    def get_roi(self) -> Optional[Tuple[int, int, int, int]]:
        return self._roi
```

设计要点：

- **完全不依赖 `cv2.selectROI`**：避免 headless OpenCV 导致的弹窗失败；EXE 自带 Qt 即可运行。
- **坐标映射**：显示使用 `KeepAspectRatio`，因此必须把“窗口坐标系的矩形”映射回“原图坐标系的 ROI”。

---

#### 22.8 GUI：任务1/任务2后台线程（避免 UI 卡死）与 GPU 开关

文件：`gui_app.py`（节选）

```python
class Worker(QtCore.QThread):
    def run(self) -> None:
        _maybe_set_offline_env_defaults()
        if self.kind == "task1":
            # 任务1 GPU 为可选：需要 onnxruntime-gpu + CUDA/cuDNN/VC++ DLL 依赖齐全
            _set_insightface_force_cpu(force_cpu=not bool(self.t1.prefer_gpu))
            from main import load_student_list, task1_attendance_check
            # ...
        if self.kind == "task2":
            from task2_behavior_flow import Task2Config, run_task2 as run_task2_flow
            # cfg.device 来自 GUI（cpu/cuda）
            # ...
```

---

#### 22.9 EXE 打包：为什么需要把本地模块当作 datas 拷贝进去

问题背景（你遇到过）：任务1运行时报 `No module named 'face_recognition_module'`。根因是：

- 任务1依赖在函数内动态 import；
- PyInstaller 静态分析不一定能把这些“运行时才 import 的本地模块”打进包；
- 在 Windows 上 `dist` 目录又可能被锁，导致你以为重打包了，但旧包没有更新。

解决方案在 `dp_pro_gui.spec`：把关键本地 `.py` 作为 datas 直接拷贝到 `_internal` 根目录，保证运行时 `import` 能找到。

文件：`dp_pro_gui.spec`（节选）

```python
hiddenimports = [
    "ultralytics",
    "cv2",
    "onnxruntime",
    "insightface",
    "face_recognition_module",
    "attendance_checker",
    "task2_behavior_flow",
    "behavior_rules_yolo",
    "preprocess_photos_insightface",
    "prepare_data",
    "app_paths",
]

datas = [
    ("assets", "assets"),
    ("README.md", "."),
    # 本地单文件模块：作为 data 直接拷贝到 _internal 根目录，确保运行时 import 可用
    ("face_recognition_module.py", "."),
    ("attendance_checker.py", "."),
    ("task2_behavior_flow.py", "."),
    ("behavior_rules_yolo.py", "."),
    ("preprocess_photos_insightface.py", "."),
    ("prepare_data.py", "."),
    ("main.py", "."),
    ("app_paths.py", "."),
]
```

---

#### 22.10 一键打包脚本（防 WinError 5、禁止 headless 回流）

文件：`build_gui_exe.bat`（节选）

```bat
chcp 65001 >nul
taskkill /F /IM DP_PRO_GUI.exe >nul 2>nul

if exist "dist\DP_PRO_GUI" (
  rmdir /S /Q "dist\DP_PRO_GUI" >nul 2>nul
)
if exist "build\dp_pro_gui" (
  rmdir /S /Q "build\dp_pro_gui" >nul 2>nul
)

python -m pip install -r requirements.txt

REM 防止依赖链把 opencv-python-headless 装回来（会导致 OpenCV HighGUI 失效）
python -m pip uninstall -y opencv-python-headless opencv-contrib-python-headless >nul 2>nul
python -m pip install --upgrade --force-reinstall opencv-python==4.8.1.78 opencv-contrib-python==4.8.1.78 --no-deps

python prepare_offline_assets.py
python -m PyInstaller -y dp_pro_gui.spec
```

工程解释：

- **kill 旧 EXE + 清理 dist/build**：避免 Windows 文件锁导致 PyInstaller 无法覆盖旧产物，从而“修了代码但 EXE 还是旧的”。
- **卸载 headless + 强制装 GUI OpenCV（--no-deps）**：避免 `insightface -> albumentations` 依赖链把 headless 装回来。
- **离线 assets**：把 `.pt` 权重与 InsightFace 模型一起打包，保证无网络可运行。

---

### 23. 进一步“越多越好”的扩展建议（如果你还要继续加长）

如果你需要把文档扩展到更偏“论文/项目书”风格（比如 2 万字以上），建议补充的章节方向：

- **实验设计与评估指标**：
  - 任务1：出勤识别的 Precision/Recall、误识别案例分析、不同时间点/参数对结果的敏感性
  - 任务2：按人/按分钟的一致性、低头类别的误报/漏报分析、与人工抽检对齐
- **数据与隐私合规**：
  - 个人信息（姓名/学号/人脸）处理与脱敏策略
  - 输出文件的访问控制建议
- **跨教室迁移**：
  - seat_map 的可复用性（同教室不同天、不同班次）
  - 相机角度变化对 desk ROI 的影响与参数模板化
- **训练路线落地（若未来要做 6 类/更多类）**：
  - 标注规范、类别定义、困难样本采样策略、训练超参、部署优化


