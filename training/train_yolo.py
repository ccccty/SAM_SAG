"""YOLO 微调脚本(Ultralytics)。

用法:
    python training/train_yolo.py --data path/to/dataset.yaml
    python training/train_yolo.py --data dataset.yaml --epochs 100 --imgsz 640 --batch 8

行为:
- 基于 weights/yolo11n.pt(缺失时回退为 ultralytics 自动下载的官方权重)微调;
- 训练产物写入 runs/yolo_<时间戳>/;
- 训练结束后把 best.pt 复制到 weights/yolo11n.pt(线上服务权重),
  可用 --deploy-name 改名或 --no-deploy 跳过。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = BASE_DIR / "weights"
RUNS_DIR = BASE_DIR / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO 微调并部署 best.pt 到 weights/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Ultralytics 数据集 yaml 路径")
    parser.add_argument("--weights", default=str(WEIGHTS_DIR / "yolo11n.pt"), help="起始权重")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸(8GB 显存建议 640)")
    parser.add_argument("--batch", type=int, default=8, help="batch(8GB 显存建议 8)")
    parser.add_argument("--device", default="auto", help="auto / cpu / 0 / 0,1 ...")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default=None, help="runs/ 下的实验名, 默认 yolo_<时间戳>")
    parser.add_argument("--deploy-name", default="yolo11n.pt", help="best.pt 复制到 weights/ 后的文件名")
    parser.add_argument("--no-deploy", action="store_true", help="不复制 best.pt 到 weights/")
    return parser.parse_args()


def resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main() -> int:
    args = parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.is_file():
        print(f"[错误] 数据集配置不存在: {data_yaml}", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[错误] ultralytics 未安装, 请先 pip install -r requirements.txt", file=sys.stderr)
        return 2

    # 起始权重: 本地缺失时用官方权重名, 交给 ultralytics 自动下载
    weights = args.weights
    if not Path(weights).is_file():
        fallback = Path(weights).name
        print(f"[提示] 本地权重 {weights} 不存在, 改用官方预训练权重 {fallback}(自动下载)")
        weights = fallback

    exp_name = args.name or f"yolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[SAM_SAG] 开始微调: data={data_yaml} weights={weights} "
          f"imgsz={args.imgsz} batch={args.batch} epochs={args.epochs}")
    model = YOLO(weights)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=resolve_device(args.device),
        workers=args.workers,
        project=str(RUNS_DIR),
        name=exp_name,
        exist_ok=False,
        verbose=True,
    )

    save_dir = Path(getattr(results, "save_dir", RUNS_DIR / exp_name))
    best_pt = save_dir / "weights" / "best.pt"
    print(f"[SAM_SAG] 训练完成, 产物目录: {save_dir}")

    if not best_pt.is_file():
        print(f"[警告] 未找到 {best_pt}, 跳过硬权重部署", file=sys.stderr)
        return 1

    if not args.no_deploy:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        target = WEIGHTS_DIR / args.deploy_name
        shutil.copy2(best_pt, target)
        print(f"[SAM_SAG] 已部署: {best_pt} → {target}")
        print("[SAM_SAG] 重启后端后 /api/detect 即使用新权重。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
