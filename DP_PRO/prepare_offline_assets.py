from __future__ import annotations

"""
把任务1/2所需的模型权重下载到本地 assets/，用于“离线可用 + 打包进exe”。

用法（在 C:\\DP_PRO 下执行）：
  python prepare_offline_assets.py
"""

import os
import shutil
from typing import Optional


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _copytree(src: str, dst: str) -> None:
    if os.path.isdir(dst):
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)


def _download_ultralytics_weights(dst_dir: str) -> None:
    # 说明：YOLO("xxx.pt") 如果本地不存在会自动下载到当前工作目录或 Ultralytics 缓存。
    # 我们在这里显式触发一次，并把 .pt 拷到 assets/weights 里，保证离线。
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError(f"未安装 ultralytics，无法准备任务2权重：{e}") from e

    _safe_mkdir(dst_dir)
    need = ["yolov8s-pose.pt", "rtdetr-l.pt"]
    for name in need:
        print(f"[assets] 准备权重: {name}")
        m = YOLO(name)  # 触发下载/加载
        src = getattr(m, "ckpt_path", None) or ""
        if not src or not os.path.exists(src):
            # 有些版本不暴露 ckpt_path，这里回退：同目录找文件
            if os.path.exists(name):
                src = os.path.abspath(name)
        if not src or not os.path.exists(src):
            raise RuntimeError(f"未能定位已下载的权重文件: {name}（Ultralytics 版本可能变化）")
        dst = os.path.join(dst_dir, name)
        shutil.copy2(src, dst)
        print(f"[assets] ✅ 已保存: {dst}")


def _prepare_insightface_models(dst_dir: str) -> None:
    # InsightFace 默认会下载到 ~/.insightface/models
    # 我们初始化 FaceAnalysis(name='buffalo_l') 触发下载，然后把整个 ~/.insightface/models 拷到 assets/insightface/models
    try:
        from insightface.app import FaceAnalysis  # type: ignore
    except Exception as e:
        raise RuntimeError(f"未安装 insightface，无法准备任务1模型：{e}") from e

    _safe_mkdir(dst_dir)
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    # prepare 会触发模型检查/下载（若缺失会联网拉取）
    app.prepare(ctx_id=-1, det_size=(640, 640))

    home = os.path.expanduser("~")
    src_models = os.path.join(home, ".insightface", "models")
    if not os.path.isdir(src_models):
        raise RuntimeError(f"未找到 InsightFace 模型目录: {src_models}（是否下载失败？）")
    _copytree(src_models, dst_dir)
    print(f"[assets] ✅ InsightFace 模型已拷贝到: {dst_dir}")


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    assets = os.path.join(root, "assets")
    weights = os.path.join(assets, "weights")
    ins_models = os.path.join(assets, "insightface", "models")

    _safe_mkdir(assets)
    print("[assets] 开始准备离线资源（首次可能需要联网下载）")
    _download_ultralytics_weights(weights)
    _prepare_insightface_models(ins_models)
    print("[assets] ✅ 全部完成。接下来可用 PyInstaller 打包 GUI EXE。")


if __name__ == "__main__":
    main()


