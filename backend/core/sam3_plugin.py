"""SAM3 文本提示分割引擎封装(真实推理后端).

设计要点:
- 可用性看"权重是否下载完整", 而不是"文件是否存在": safetensors 头部记录了每个张量的
  data_offsets, 只有 8 字节头长 + JSON 头长 + 数据区长度 <= 文件大小时才算就绪,
  这样下载到一半的 model.safetensors 不会被误判为可用(契约: 权重缺失 → 503);
- 懒加载 + 进程内单例缓存: 首次 /api/segment 才 import torch/transformers 并加载权重,
  服务启动与 /api/health /api/models 都不占显存; CUDA 上用 fp16 省显存(8GB 卡);
- 多概念: 视觉特征(Sam3Model.get_vision_features)对同一张图只算一次并复用, 每个概念
  单独跑一次文本条件前向(逗号分隔的整串当短语会语义跑偏), 实例带 concept 标记;
- 输出严格按 docs/API-sam3.md §3: instances[...] + 单实例黑白 mask PNG + 多实例彩色
  mask PNG + 原图半透明叠加 PNG, 全部 base64(PIL 编码, 与 inspector._png_base64 同风格);
- 异常映射: ModelNotReadyError → 503 MODEL_NOT_READY; SegmentValidationError →
  400 VALIDATION_ERROR; CUDA OOM → RuntimeError(可读原因) → 500 INTERNAL_ERROR。
"""

from __future__ import annotations

import base64
import io
import json
import math
import struct
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from . import config
from .detector import ModelNotReadyError

# ---------------------------------------------------------------------------
# 契约常量(docs/API-sam3.md §2/§3)
# ---------------------------------------------------------------------------
SAM3_WEIGHT_NAME = "model.safetensors"
SAM3_MODEL_ID = "sam3.1-fp16"
SAM3_CONCEPTS_HINT = ["bottle", "screw", "scratch", "broken", "dent"]

# 加载前要求的最小可用显存(权重 1.6GB + 1008x1008 前向激活), 低于此值直接给可读报错
_MIN_FREE_VRAM_MB = 2200

# 实例配色(高对比, 循环使用)
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 64, 64),
    (64, 200, 255),
    (255, 210, 64),
    (110, 255, 110),
    (220, 120, 255),
    (255, 140, 40),
    (64, 255, 210),
    (255, 64, 180),
    (150, 150, 255),
    (200, 255, 64),
)

_OVERLAY_ALPHA = 0.45
_BOX_WIDTH = 3


class SegmentValidationError(ValueError):
    """请求参数或图像不合法 → API 层转 400 VALIDATION_ERROR。"""


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _png_base64(rgb: np.ndarray) -> str:
    """(H, W, 3) 数组 → base64 PNG 字符串(防御性转 uint8, 与 inspector 同风格)。"""
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mask_png_base64(mask: np.ndarray) -> str:
    """单实例 0/1 掩码 → 黑白 RGB PNG(白=命中)。"""
    binary = (mask > 0).astype(np.uint8) * 255
    return _png_base64(np.repeat(binary[:, :, None], 3, axis=2))


def _color_mask(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """0/1 掩码 + 颜色 → (H, W, 3) 彩色平铺图(未命中为全黑)。"""
    canvas = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    hit = mask.astype(bool)
    canvas[hit] = np.array(color, dtype=np.uint8)
    return canvas


def _parse_concepts(concepts: Any) -> list[str]:
    """把 "bottle,broken" / ["bottle", "broken"] 统一成去重后的概念列表。"""
    if isinstance(concepts, str):
        raw = concepts.split(",")
    else:
        raw = []
        for item in list(concepts or []):
            raw.extend(str(item).split(","))
    seen: list[str] = []
    for token in raw:
        name = token.strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def _safetensors_expected_size(path: Path) -> int:
    """按 safetensors 头部推算完整文件应有的大小; 头部不完整时抛 ValueError。"""
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            raise ValueError("权重文件不足 8 字节, 还没有写入头部")
        (header_len,) = struct.unpack("<Q", raw)
        if header_len <= 1 or header_len > 64 * 1024 * 1024:
            raise ValueError(f"权重文件头长度异常({header_len}), 可能仍是半成品")
        header = fh.read(header_len)
    if len(header) < header_len:
        raise ValueError("权重文件头未下载完整")
    try:
        meta = json.loads(header)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"权重文件头 JSON 无法解析: {exc}") from exc
    data_end = 0
    for key, info in meta.items():
        if key == "__metadata__":
            continue
        offsets = info.get("data_offsets") if isinstance(info, dict) else None
        if not (isinstance(offsets, list) and len(offsets) == 2):
            raise ValueError(f"权重张量 {key} 缺少 data_offsets")
        data_end = max(data_end, int(offsets[1]))
    return 8 + header_len + data_end


def _human_mb(num_bytes: float) -> str:
    return f"{num_bytes / (1024 * 1024):.0f}MB"


# ---------------------------------------------------------------------------
# SAM3 插件
# ---------------------------------------------------------------------------
class Sam3Plugin:
    """SAM3 插件门面: 权重完整性探测 + 懒加载单例 + 文本提示实例分割。"""

    def __init__(self, weights_dir: Path | None = None) -> None:
        self.weights_dir = weights_dir or config.SAM3_WEIGHTS_DIR
        self._model: Any = None
        self._processor: Any = None
        self._device: str | None = None
        self._dtype: Any = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        # 权重完整性缓存: (size, mtime) → (ready, detail); 下载过程中每次都会变化,
        # 所以缓存不会把"还没下完"的状态钉死。
        self._integrity: tuple[tuple[int, float], bool, str] | None = None

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def weights_path(self) -> Path:
        return self.weights_dir / SAM3_WEIGHT_NAME

    def _check_weights(self) -> tuple[bool, str]:
        """返回 (是否就绪, 说明)。不会抛异常。"""
        path = self.weights_path
        try:
            stat = path.stat()
        except OSError:
            return False, f"未找到权重文件 {path}"
        stamp = (stat.st_size, stat.st_mtime)
        if self._integrity is not None and self._integrity[0] == stamp:
            return self._integrity[1], self._integrity[2]
        try:
            expected = _safetensors_expected_size(path)
        except (OSError, ValueError) as exc:
            ready, detail = False, f"{SAM3_WEIGHT_NAME} 未就绪: {exc}"
        else:
            if stat.st_size < expected:
                ready = False
                detail = (
                    f"{SAM3_WEIGHT_NAME} 尚未下载完整"
                    f"({_human_mb(stat.st_size)}/{_human_mb(expected)}), 下载完成后自动可用"
                )
            else:
                ready, detail = True, _human_mb(stat.st_size)
        self._integrity = (stamp, ready, detail)
        return ready, detail

    @property
    def available(self) -> bool:
        return self._check_weights()[0]

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def status(self) -> dict:
        """GET /api/models 里的 sam3 段(字段见契约 §2, 附带少量调试字段)。"""
        ready, detail = self._check_weights()
        weights_mb = 0
        try:
            weights_mb = int(self.weights_path.stat().st_size // (1024 * 1024))
        except OSError:
            pass
        if ready:
            message = f"sam3.1 fp16 就绪 ({_human_mb(weights_mb * 1024 * 1024)})"
        else:
            message = f"unavailable: {detail}"
        return {
            "available": ready,
            "message": message,
            "model": SAM3_MODEL_ID,
            "weights_mb": weights_mb,
            "concepts_hint": list(SAM3_CONCEPTS_HINT),
            # 以下为辅助字段(前端徽章只读 message, 多出的键不影响既有契约)
            "loaded": self.loaded,
            "device": self._device,
            "weights_dir": str(self.weights_dir),
            "weights": SAM3_WEIGHT_NAME if self.weights_path.is_file() else None,
            "capabilities": ["text_prompt_instance_segmentation"] if ready else [],
        }

    # ------------------------------------------------------------------
    # 懒加载
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """首次调用时加载模型; 已在内存则直接返回。失败一律转 ModelNotReadyError。"""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            ready, detail = self._check_weights()
            if not ready:
                raise ModelNotReadyError(f"SAM3 不可用: {detail}")
            try:
                import torch
                from transformers import Sam3Model, Sam3Processor
            except ImportError as exc:  # 依赖缺失与权重缺失对调用方一样是"未就绪"
                raise ModelNotReadyError(
                    f"SAM3 依赖未安装({exc}), 需要 .venv 内 transformers>=5.21"
                ) from exc

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            if device == "cuda":
                free, _total = torch.cuda.mem_get_info()
                if free < _MIN_FREE_VRAM_MB * 1024 * 1024:
                    raise ModelNotReadyError(
                        f"SAM3 加载需要约 {_MIN_FREE_VRAM_MB}MB 可用显存, "
                        f"当前仅 {free / (1024 * 1024):.0f}MB; 请关闭占用显存的进程后重试"
                    )
            started = time.perf_counter()
            try:
                model = Sam3Model.from_pretrained(str(self.weights_dir), dtype=dtype)
                model.to(device).eval()
                processor = Sam3Processor.from_pretrained(str(self.weights_dir))
            except Exception as exc:  # noqa: BLE001 - 权重损坏/半成品/显存不足都要转 503
                self._model = self._processor = None
                if device == "cuda":
                    torch.cuda.empty_cache()
                raise ModelNotReadyError(
                    f"SAM3 权重加载失败(可能仍在下载/文件损坏): {exc}"
                ) from exc
            self._model = model
            self._processor = processor
            self._device = device
            self._dtype = dtype
            self._load_ms = round((time.perf_counter() - started) * 1000.0, 1)

    def _to_device(self, inputs: dict) -> dict:
        """BatchEncoding → 按需搬到 device, 浮点转 fp16, 其余保持整型。"""
        import torch

        moved = {}
        for key, value in dict(inputs).items():
            if not torch.is_tensor(value):
                moved[key] = value
            elif torch.is_floating_point(value):
                moved[key] = value.to(self._device, dtype=self._dtype)
            else:
                moved[key] = value.to(self._device)
        return moved

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def segment(
        self,
        image: np.ndarray,
        concepts: Any,
        threshold: float = config.SAM3_DEFAULT_THRESHOLD,
        mask_threshold: float = config.SAM3_DEFAULT_MASK_THRESHOLD,
    ) -> dict:
        """文本条件实例分割, 返回契约 §3 的 result 对象。"""
        image = self._validate_image(image)
        names = _parse_concepts(concepts)
        if not names:
            raise SegmentValidationError("缺少概念提示 text, 例如 'bottle,broken'")
        threshold = self._validate_threshold("threshold", threshold)
        mask_threshold = self._validate_threshold("mask_threshold", mask_threshold)

        self._ensure_loaded()
        import torch

        pil_image = Image.fromarray(image)
        height, width = image.shape[0], image.shape[1]
        started = time.perf_counter()
        instances: list[dict] = []
        with self._infer_lock, torch.inference_mode():
            try:
                pixel_inputs = self._to_device(self._processor(images=pil_image, return_tensors="pt"))
                # 视觉特征对同一张图只算一次, 多概念复用(省下每个概念的 ViT 前向)
                vision_embeds = self._model.get_vision_features(pixel_values=pixel_inputs["pixel_values"])
                for name in names:
                    text_inputs = self._to_device(self._processor(text=name, return_tensors="pt"))
                    outputs = self._model(
                        vision_embeds=vision_embeds,
                        input_ids=text_inputs["input_ids"],
                        attention_mask=text_inputs.get("attention_mask"),
                    )
                    per_image = self._processor.post_process_instance_segmentation(
                        outputs,
                        threshold=threshold,
                        mask_threshold=mask_threshold,
                        target_sizes=[(height, width)],
                    )
                    instances.extend(self._to_instances(name, per_image[0]))
            except torch.cuda.OutOfMemoryError as exc:
                if self._device == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA 显存不足, 无法完成分割: {exc}; 可尝试缩小输入图片后重试"
                ) from exc
        inference_ms = round((time.perf_counter() - started) * 1000.0, 1)

        instances.sort(key=lambda item: item["score"], reverse=True)
        masks = [item.pop("_mask") for item in instances]
        for item, mask in zip(instances, masks):  # 契约 §3: 每个实例附独立 mask PNG
            item["mask_png_base64"] = _mask_png_base64(mask)
        return {
            "concepts": names,
            "model": SAM3_MODEL_ID,
            "device": self._device or "cpu",
            "inference_ms": inference_ms,
            "image_size": [width, height],
            "instances": instances,
            "mask_png_base64": self._compose_mask_png(masks, height, width),
            "overlay_png_base64": self._compose_overlay_png(image, instances, masks),
        }

    @staticmethod
    def _validate_image(image: Any) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise SegmentValidationError("图像必须是 RGB uint8 数组")
        if image.ndim != 3 or image.shape[2] != 3:
            raise SegmentValidationError(f"图像形状不合法: {getattr(image, 'shape', None)}, 需要 (H, W, 3)")
        if image.shape[0] < 8 or image.shape[1] < 8:
            raise SegmentValidationError(f"图像尺寸过小: {image.shape[1]}x{image.shape[0]}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    @staticmethod
    def _validate_threshold(name: str, value: Any) -> float:
        try:
            num = float(value)
        except (TypeError, ValueError) as exc:
            raise SegmentValidationError(f"{name} 必须是浮点数, 收到 {value!r}") from exc
        if not math.isfinite(num) or not (0.05 <= num <= 0.95):
            raise SegmentValidationError(f"{name} 超出范围 [0.05, 0.95]: {num}")
        return num

    @staticmethod
    def _to_instances(concept: str, result: dict) -> list[dict]:
        """post_process 的单图结果 → 契约里的实例条目(暂存 _mask 供编码)。"""
        import torch

        scores = result["scores"]
        boxes = result["boxes"]
        masks = result["masks"]
        if not torch.is_tensor(scores) or scores.numel() == 0:
            return []
        scores_np = scores.detach().float().cpu().numpy()
        boxes_np = boxes.detach().float().cpu().numpy()
        masks_np = masks.detach().to(torch.uint8).cpu().numpy()
        height, width = masks_np.shape[1], masks_np.shape[2]
        total_px = float(max(height * width, 1))
        items: list[dict] = []
        for idx in range(scores_np.shape[0]):
            mask = masks_np[idx]
            area_px = int(mask.sum())
            if area_px <= 0:  # 空掩码无法展示, 直接丢弃
                continue
            items.append(
                {
                    "concept": concept,
                    "score": round(float(scores_np[idx]), 4),
                    "box": [round(float(v), 1) for v in boxes_np[idx][:4]],
                    "area_px": area_px,
                    "area_ratio": round(area_px / total_px, 4),
                    "_mask": mask,
                }
            )
        return items

    @staticmethod
    def _compose_mask_png(masks: list[np.ndarray], height: int, width: int) -> str:
        """所有实例的彩色叠加图(每实例一色, 背景黑)。"""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for idx, mask in enumerate(masks):
            color = _PALETTE[idx % len(_PALETTE)]
            hit = mask.astype(bool)
            canvas[hit] = np.array(color, dtype=np.uint8)
        return _png_base64(canvas)

    @staticmethod
    def _compose_overlay_png(image: np.ndarray, instances: list[dict], masks: list[np.ndarray]) -> str:
        """原图 + 半透明彩色掩码 + 实例框。"""
        overlay = image.astype(np.float32)
        for idx, mask in enumerate(masks):
            color = np.array(_PALETTE[idx % len(_PALETTE)], dtype=np.float32)
            hit = mask.astype(bool)
            overlay[hit] = overlay[hit] * (1.0 - _OVERLAY_ALPHA) + color * _OVERLAY_ALPHA
        canvas = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(canvas)
        for idx, item in enumerate(instances):
            color = _PALETTE[idx % len(_PALETTE)]
            x1, y1, x2, y2 = (float(v) for v in item["box"])
            draw.rectangle([x1, y1, x2, y2], outline=color, width=_BOX_WIDTH)
        return _png_base64(np.asarray(canvas, dtype=np.uint8))

    # ------------------------------------------------------------------
    # 兼容旧接口(前端/脚本可能仍调用)
    # ------------------------------------------------------------------
    def segment_with_text(self, image: np.ndarray, text_prompt: str) -> dict:
        """旧签名薄包装: 返回 {count, inference_ms, masks_png_base64}。"""
        result = self.segment(image, text_prompt)
        return {
            "count": len(result["instances"]),
            "inference_ms": result["inference_ms"],
            "masks_png_base64": [result["mask_png_base64"]],
        }


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------
_plugin: Sam3Plugin | None = None


def get_sam3_plugin() -> Sam3Plugin:
    global _plugin
    if _plugin is None:
        _plugin = Sam3Plugin()
    return _plugin
