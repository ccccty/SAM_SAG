"""SAM3 文本提示分割自测脚本(真实推理, 端到端跑一次并落盘 PNG)。

用法:
    .venv\Scripts\python.exe scripts\smoke_segment.py
    .venv\Scripts\python.exe scripts\smoke_segment.py --text "bottle,broken" --threshold 0.3

行为:
- 默认取 data/mvtec/bottle/test/broken_large/000.png, 不存在则取 data/mvtec 下任一测试图;
- 权重未就绪时打印原因并以退出码 2 结束(便于 CI 区分"没权重"和"跑挂了");
- 跑通后打印 实例数 / 每个实例的 concept-score-box-area / 耗时打点 / 显存占用,
  并把彩色 mask 与叠加图写到 runs/sam3_smoke_*.png。
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from PIL import Image

from backend.core import config
from backend.core.detector import ModelNotReadyError
from backend.core.sam3_plugin import get_sam3_plugin

DEFAULT_IMAGE = config.MVTEC_DIR / "bottle" / "test" / "broken_large" / "000.png"


def _pick_image(explicit: str | None) -> Path:
    """优先用显式路径, 其次默认 bottle 缺陷图, 最后在 data/mvtec 里找第一张测试图。"""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"[FAIL] 指定的图片不存在: {path}")
        return path
    if DEFAULT_IMAGE.is_file():
        return DEFAULT_IMAGE
    for pattern in ("*/test/*/*.png", "*/test/*/*.jpg", "*/test/*/*.bmp"):
        found = sorted(config.MVTEC_DIR.glob(pattern))
        if found:
            return found[0]
    raise SystemExit(f"[FAIL] 在 {config.MVTEC_DIR} 下找不到任何测试图片")


def _save(base64_png: str, target: Path) -> int:
    raw = base64.b64decode(base64_png)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return len(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SAM3 文本提示分割冒烟测试")
    parser.add_argument("--image", default=None, help="输入图片路径")
    parser.add_argument("--text", default="bottle,broken", help="逗号分隔的概念提示")
    parser.add_argument("--threshold", type=float, default=config.SAM3_DEFAULT_THRESHOLD)
    parser.add_argument("--mask-threshold", type=float, default=config.SAM3_DEFAULT_MASK_THRESHOLD)
    args = parser.parse_args(argv)

    image_path = _pick_image(args.image)
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    plugin = get_sam3_plugin()
    status = plugin.status()

    print("=" * 68)
    print("SAM3 冒烟测试")
    print(f"  解释器   : {sys.executable}")
    print(f"  图片     : {image_path}  ({image.shape[1]}x{image.shape[0]})")
    print(f"  概念     : {args.text}   阈值={args.threshold} mask阈值={args.mask_threshold}")
    print(f"  权重组件 : available={status['available']} weights_mb={status['weights_mb']}")
    print(f"  状态说明 : {status['message']}")
    try:
        import torch

        print(f"  torch    : {torch.__version__}  cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(
                f"  GPU      : {torch.cuda.get_device_name(0)}  "
                f"空闲 {free / 1048576:.0f}MB / 共 {total / 1048576:.0f}MB"
            )
    except Exception as exc:  # noqa: BLE001 - 仅用于诊断打印
        print(f"  torch    : 不可用 ({exc})")
    print("=" * 68)

    if not status["available"]:
        print("[NOT-READY] 权重未就绪, 按契约应返回 503 MODEL_NOT_READY, 冒烟测试跳过推理。")
        return 2

    started = time.perf_counter()
    try:
        result = plugin.segment(
            image,
            args.text,
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
        )
    except ModelNotReadyError as exc:
        print(f"[NOT-READY] {exc}")
        return 2
    total_ms = (time.perf_counter() - started) * 1000.0

    instances = result["instances"]
    print(
        f"device={result['device']} model={result['model']} "
        f"image_size={result['image_size']} 概念={result['concepts']}"
    )
    print(f"实例数: {len(instances)}")
    for idx, item in enumerate(instances):
        box = ", ".join(f"{v:.1f}" for v in item["box"])
        print(
            f"  #{idx} concept={item['concept']:<10} score={item['score']:.4f} "
            f"box=[{box}] area_px={item['area_px']} area_ratio={item['area_ratio']}"
        )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    mask_bytes = _save(result["mask_png_base64"], config.RUNS_DIR / f"sam3_smoke_mask_{stamp}.png")
    overlay_bytes = _save(result["overlay_png_base64"], config.RUNS_DIR / f"sam3_smoke_overlay_{stamp}.png")
    print(
        f"保存: runs/sam3_smoke_mask_{stamp}.png ({mask_bytes} B), "
        f"runs/sam3_smoke_overlay_{stamp}.png ({overlay_bytes} B)"
    )

    print("-" * 68)
    print(f"  推理耗时 inference_ms = {result['inference_ms']}")
    print(f"  端到端耗时 total_ms   = {total_ms:.1f}")
    try:
        import torch

        if torch.cuda.is_available():
            print(
                f"  显存 allocated = {torch.cuda.max_memory_allocated() / 1048576:.0f}MB "
                f"reserved = {torch.cuda.max_memory_reserved() / 1048576:.0f}MB"
            )
    except Exception:  # noqa: BLE001 - 显存统计只是附加信息
        pass
    print("-" * 68)

    if not instances:
        print("[FAIL] 实例数 0, 未能命中任何候选")
        return 1
    if mask_bytes <= 0 or overlay_bytes <= 0:
        print("[FAIL] 生成的 PNG 为空")
        return 1
    print("[OK] 冒烟测试通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
