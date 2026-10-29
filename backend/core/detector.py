"""YOLO 目标检测引擎封装.

设计要点:
- 懒加载: 首次推理时才 import ultralytics 并加载权重, 服务启动快;
- 优雅降级: 权重缺失或依赖未安装时不抛栈, status()/available 反映为不可用,
  API 层据此返回 503;
- 线程安全: 加载与推理解耦, 加载加锁防止并发重复加载。
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from . import config


class ModelNotReadyError(RuntimeError):
    """模型文件缺失 / 依赖未安装时抛出, 由 API 层转换为 503。"""


class YoloDetector:
    """Ultralytics YOLO 的薄封装。"""

    def __init__(self, weights: "config.Path | None" = None) -> None:
        self.weights_path = weights or config.YOLO_WEIGHTS
        self._model: Any = None
        self._device: str | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def weights_exists(self) -> bool:
        return self.weights_path.is_file()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def available(self) -> bool:
        """可用 = 权重在盘上。依赖是否可导入在 load() 时确认。"""
        return self.weights_exists

    def status(self) -> dict:
        if self.loaded:
            message = "已加载, 就绪"
        elif self.weights_exists:
            message = "权重已就绪(首次检测时加载)"
        else:
            message = f"模型未就绪: 缺少权重 {self.weights_path.name}, 请运行 scripts/download_assets.py --yolo"
        return {
            "available": self.available,
            "loaded": self.loaded,
            "weights": str(self.weights_path),
            "device": self._device,
            "message": message,
        }

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载权重到内存/显存。线程安全, 重复调用为幂等。"""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:  # double-checked
                return
            if not self.weights_exists:
                raise ModelNotReadyError(
                    f"YOLO 模型未就绪: 找不到权重文件 {self.weights_path}"
                )
            try:
                from ultralytics import YOLO  # 延迟导入, 避免拖累服务启动
            except ImportError as exc:  # pragma: no cover - 环境缺包时
                raise ModelNotReadyError(
                    "ultralytics 未安装, 请在 .venv 中执行 pip install -r requirements.txt"
                ) from exc

            self._model = YOLO(str(self.weights_path))
            # 记录实际运行设备(cuda:0 / cpu)
            try:
                self._device = str(self._model.device)
            except Exception:  # noqa: BLE001 - 仅为状态展示, 不影响推理
                self._device = None

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray, conf: float | None = None) -> tuple[list[dict], float]:
        """对一张 RGB 图像(H, W, 3)做检测。

        返回 (detections, inference_ms); detections 元素为
        {"label": str, "confidence": float, "box": [x1, y1, x2, y2]}(原图像素坐标)。
        """
        self.load()
        assert self._model is not None

        conf_thres = config.DEFAULT_CONF_THRESHOLD if conf is None else float(conf)
        start = time.perf_counter()
        results = self._model.predict(image, conf=conf_thres, verbose=False)
        inference_ms = (time.perf_counter() - start) * 1000.0

        detections: list[dict] = []
        if not results:
            return detections, inference_ms

        result = results[0]
        names = result.names or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections, inference_ms

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), score, cls_id in zip(xyxy, scores, classes):
            detections.append(
                {
                    "label": str(names.get(int(cls_id), f"class_{cls_id}")),
                    "confidence": round(float(score), 4),
                    "box": [
                        round(float(x1), 1),
                        round(float(y1), 1),
                        round(float(x2), 1),
                        round(float(y2), 1),
                    ],
                }
            )
        # 置信度从高到低排序, 前端展示更直观
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections, inference_ms


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------
_detector: YoloDetector | None = None


def get_detector() -> YoloDetector:
    global _detector
    if _detector is None:
        _detector = YoloDetector()
    return _detector
