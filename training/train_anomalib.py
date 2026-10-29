"""在 MVTec AD 数据集上训练 anomalib 瑕疵检测模型(EfficientAD / PatchCore)。

用法:
    python training/train_anomalib.py --category bottle
    python training/train_anomalib.py --category bottle --model patchcore --epochs 30

行为:
- 数据集默认读取 data/mvtec/<category>(标准 MVTec 目录结构);
- checkpoint 写入 weights/anomalib/<category>/, 文件名带模型与时间戳;
- 同步写入 threshold.json(从测试集阈值读取, 失败则写默认值 0.5);
- 默认参数面向 8GB 显存: image_size 256, batch 4, num_workers 0(Windows 友好)。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# 项目根目录(脚本位于 training/ 下)
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = BASE_DIR / "data" / "mvtec"
DEFAULT_OUT_ROOT = BASE_DIR / "weights" / "anomalib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练 anomalib 瑕疵检测模型并导出 ckpt 到 weights/anomalib/<category>/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--category", required=True, help="MVTec 类别名, 如 bottle / cable / screw")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="MVTec 数据集根目录")
    parser.add_argument("--model", default="efficientad", choices=["efficientad", "patchcore"], help="算法")
    parser.add_argument("--image-size", type=int, default=256, help="输入边长(8GB 显存建议 256)")
    parser.add_argument("--batch-size", type=int, default=4, help="训练 batch(EfficientAD 建议 1~4)")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 进程数(Windows 建议 0)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="训练设备")
    parser.add_argument("--skip-test", action="store_true", help="训练后不跑测试集评估")
    return parser.parse_args()


def build_datamodule(args: argparse.Namespace):
    """构建 MVTec 数据模块; 兼容 anomalib 1.x(MVTec)与 2.x(MVTecAD)命名。"""
    try:
        from anomalib.data import MVTecAD as MvtecData  # anomalib 2.x
    except ImportError:
        from anomalib.data import MVTec as MvtecData  # anomalib 1.x

    kwargs = dict(
        root=args.data_root,
        category=args.category,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    try:
        return MvtecData(image_size=(args.image_size, args.image_size), **kwargs)
    except TypeError:
        # 老版本不接受 image_size 参数, 用默认变换
        return MvtecData(**kwargs)


def build_model(args: argparse.Namespace):
    """按名称实例化模型; 兼容 anomalib 不同版本的类名大小写。"""
    import anomalib.models as models_mod

    candidates = {
        "efficientad": ("EfficientAd", "Efficientad"),
        "patchcore": ("Patchcore", "PatchCore"),
    }[args.model]
    for name in candidates:
        cls = getattr(models_mod, name, None)
        if cls is not None:
            # PatchCore 收紧 coreset 采样比例, 控制显存占用
            if args.model == "patchcore":
                try:
                    return cls(coreset_sampling_ratio=0.1)
                except TypeError:
                    return cls()
            return cls()
    raise RuntimeError(f"当前 anomalib 版本找不到模型类: {candidates}")


def resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main() -> int:
    # Windows 管道(如 Tee-Object)下 stdout 是 GBK, rich 进度条打印 '•' 会炸;
    # 统一重配为 utf-8(replace 兜底), 不影响任何训练逻辑。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass
    args = parse_args()

    category_dir = Path(args.data_root) / args.category
    if not (category_dir / "train" / "good").is_dir():
        print(f"[错误] 数据集不完整: 缺少 {category_dir / 'train' / 'good'}", file=sys.stderr)
        print("       请先运行: python training/prepare_dataset.py --category " + args.category, file=sys.stderr)
        return 2

    out_dir = DEFAULT_OUT_ROOT / args.category
    out_dir.mkdir(parents=True, exist_ok=True)

    from anomalib.engine import Engine
    # 必须用 anomalib 的 ModelCheckpoint 子类: Engine 只认这个类型,
    # 否则它会再注入一个默认 ModelCheckpoint, lightning 会因双实例报错。
    from anomalib.callbacks.checkpoint import ModelCheckpoint

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = f"{args.model}_{timestamp}"

    datamodule = build_datamodule(args)
    model = build_model(args)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(out_dir),
        filename=ckpt_name,
        save_last=True,
        save_top_k=1,
        monitor=None,          # EfficientAD 无可用验证指标, 保留最后轮次
        auto_insert_metric_name=False,
    )

    engine = Engine(
        max_epochs=args.epochs,
        accelerator=resolve_device(args.device),
        devices=1,
        callbacks=[checkpoint_cb],
        default_root_dir=str(BASE_DIR / "runs" / "anomalib" / args.category),
    )

    print(f"[SAM_SAG] 开始训练: 类别={args.category} 模型={args.model} "
          f"image_size={args.image_size} batch={args.batch_size} epochs={args.epochs}")
    engine.fit(model=model, datamodule=datamodule)

    # 找最终 ckpt: ModelCheckpoint 可能生成 <name>.ckpt / <name>-last.ckpt
    ckpts = sorted(out_dir.glob(ckpt_name + "*.ckpt"), key=lambda p: p.stat().st_mtime)
    best_ckpt = ckpts[-1] if ckpts else None
    print(f"[SAM_SAG] 训练完成, 权重: {best_ckpt or '(未找到, 请检查日志)'}")

    # 测试集评估 + 阈值落盘(失败不阻断训练产物)
    threshold = 0.5
    if not args.skip_test and best_ckpt is not None:
        try:
            results = engine.test(model=model, datamodule=datamodule, ckpt_path=str(best_ckpt))
            print(f"[SAM_SAG] 测试集指标: {results}")
            # 从模型后处理器读归一化空间阈值(不同版本字段名不同, 尽力而为)
            for attr_chain in (
                ("image_threshold", "value"),
                ("threshold",),
            ):
                obj = model
                try:
                    for attr in attr_chain:
                        obj = getattr(obj, attr)
                    threshold = float(obj.item() if hasattr(obj, "item") else obj)
                    break
                except (AttributeError, TypeError, ValueError):
                    continue
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] 测试集评估失败(不影响权重): {exc}", file=sys.stderr)

    (out_dir / "threshold.json").write_text(
        json.dumps({"threshold": round(threshold, 4), "model": args.model,
                    "image_size": args.image_size, "trained_at": timestamp},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[SAM_SAG] 阈值与元信息已写入 {out_dir / 'threshold.json'}")
    print("[SAM_SAG] 重启后端后即可在页面选择该类别进行瑕疵检测。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
