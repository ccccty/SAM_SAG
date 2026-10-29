"""SAM3 可选插件: 文本提示分割(text-prompt segmentation)预留接口。

定位:
- SAM3 权重放入 weights/sam3/ 后, available 自动变为 True;
- 当前版本仅提供接口签名与可用性探测; 待 SAM3 官方推理库发布后,
  在 segment_with_text() 内接入真实推理, API 与前端无需改动。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .detector import ModelNotReadyError

# 认可的 SAM3 权重文件后缀
_SAM3_SUFFIXES = {".pt", ".pth", ".safetensors", ".ckpt"}


class Sam3Plugin:
    """SAM3 插件门面: 可用性检测 + 文本分割接口签名。"""

    def __init__(self, weights_dir: Path | None = None) -> None:
        self.weights_dir = weights_dir or config.SAM3_WEIGHTS_DIR
        self._model: Any = None

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    def _weight_files(self) -> list[Path]:
        if not self.weights_dir.is_dir():
            return []
        return [
            p for p in sorted(self.weights_dir.iterdir())
            if p.is_file() and p.suffix.lower() in _SAM3_SUFFIXES
        ]

    @property
    def available(self) -> bool:
        return len(self._weight_files()) > 0

    def status(self) -> dict:
        files = self._weight_files()
        return {
            "available": self.available,
            "weights_dir": str(self.weights_dir),
            "weights": [p.name for p in files],
            "loaded": self._model is not None,
            "capabilities": ["text_prompt_segmentation"] if self.available else [],
            "message": (
                "检测到 SAM3 权重, 接口预留就绪"
                if self.available
                else "unavailable: 未检测到 weights/sam3/ 权重, SAM3 为可选插件, 不影响检测与瑕疵检测功能"
            ),
        }

    # ------------------------------------------------------------------
    # 推理接口(预留)
    # ------------------------------------------------------------------
    def segment_with_text(
        self,
        image: np.ndarray,
        text_prompt: str,
    ) -> dict:
        """按文本提示分割图像。

        参数:
            image: RGB 图像, uint8 (H, W, 3)。
            text_prompt: 目标描述文本, 如 "scratch on the bottle"。

        返回(约定, 接入后不变):
            {
                "masks_png_base64": [str, ...],   # 每个目标一张 0/255 掩码 PNG
                "count": int,                     # 命中的目标数
                "inference_ms": float,
            }

        当前行为: SAM3 官方推理链路未接入, 始终抛出 ModelNotReadyError。
        """
        if not self.available:
            raise ModelNotReadyError(
                "SAM3 插件不可用: 请将权重放入 weights/sam3/ 后重启服务"
            )
        # TODO(sam3): 待 SAM3 官方发布后, 在此完成:
        #   1. 懒加载模型到 self._model;
        #   2. 以 text_prompt 为条件推理分割掩码;
        #   3. 掩码编码为 PNG base64 按上述契约返回。
        raise ModelNotReadyError(
            "SAM3 推理链路尚未接入(官方推理库待发布), 接口签名已预留: "
            "segment_with_text(image, text_prompt)"
        )


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------
_plugin: Sam3Plugin | None = None


def get_sam3_plugin() -> Sam3Plugin:
    global _plugin
    if _plugin is None:
        _plugin = Sam3Plugin()
    return _plugin
