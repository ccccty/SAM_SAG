"""权重与数据集下载引导脚本。

用法:
    python scripts/download_assets.py            # 只打印全部指引
    python scripts/download_assets.py --yolo     # 自动下载 yolo11n.pt
    python scripts/download_assets.py --mvtec    # 打印 MVTec AD 数据集获取指引

说明:
- anomalib 瑕疵模型需要按类别自行训练(见 training/train_anomalib.py), 无官方成品分发;
- SAM3 权重待官方发布, 本脚本打印放置指引即可。
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = BASE_DIR / "weights"

# ultralytics 官方 release 资产(与 ultralytics>=8.3 配套)
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
YOLO_TARGET = WEIGHTS_DIR / "yolo11n.pt"

MVTEC_GUIDE = """
MVTec AD 数据集获取指引
=========================
1. 官方页面: https://www.mvtec.com/company/research/datasets/mvtec-ad
   (需填写表单申请, 免费用于科研)
2. 解压后按类别整理为:
       data/mvtec/<category>/train/good/*.png
       data/mvtec/<category>/test/good/*.png
       data/mvtec/<category>/test/<defect_type>/*.png
       data/mvtec/<category>/ground_truth/<defect_type>/*.png   (可选)
3. 整理完成后运行:
       python training/prepare_dataset.py --category <category>
4. 训练瑕疵检测模型:
       python training/train_anomalib.py --category <category>
"""

SAM3_GUIDE = """
SAM3(可选插件)放置指引
=========================
SAM3 为预留的文本提示分割能力, 当前不影响主功能。
官方权重发布后, 放入 weights/sam3/ 目录即可被后端自动识别
(状态见 /api/models → sam3)。推理接入点: backend/core/sam3_plugin.py。
"""


def download(url: str, target: Path) -> bool:
    """带进度显示的 urllib 下载。成功返回 True。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"下载: {url}")
    print(f"  → {target}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1 << 20)  # 1MB
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    print(f"\r  {done // 1024 // 1024}MB / {total // 1024 // 1024}MB ({pct}%)", end="", flush=True)
        print()
        tmp.replace(target)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"\n[失败] {exc}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM_SAG 权重与数据集下载引导")
    parser.add_argument("--yolo", action="store_true", help="自动下载 yolo11n.pt")
    parser.add_argument("--mvtec", action="store_true", help="打印 MVTec AD 获取指引")
    args = parser.parse_args()

    ok = True

    if args.yolo:
        if YOLO_TARGET.is_file():
            print(f"[跳过] 已存在: {YOLO_TARGET}")
        else:
            ok = download(YOLO_URL, YOLO_TARGET) and ok
    else:
        print(f"YOLO 权重: 运行 'python scripts/download_assets.py --yolo' 自动下载到 {YOLO_TARGET}")
        print(f"          (或手动下载 {YOLO_URL})")
        print()

    if args.mvtec:
        print(MVTEC_GUIDE)
    elif not args.yolo:
        print(MVTEC_GUIDE)

    if not args.yolo and not args.mvtec:
        print(SAM3_GUIDE)
        print("anomalib 瑕疵模型: 无官方成品, 请用 training/train_anomalib.py 自行训练。")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
