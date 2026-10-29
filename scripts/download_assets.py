"""权重与数据集下载引导脚本。

用法:
    python scripts/download_assets.py            # 只打印全部指引
    python scripts/download_assets.py --yolo     # 自动下载 yolo11n.pt
    python scripts/download_assets.py --mvtec    # 打印 MVTec AD 数据集获取指引
    python scripts/download_assets.py --sam3     # 自动下载 SAM3 权重(约 1.6GB)

说明:
- anomalib 瑕疵模型需要按类别自行训练(见 training/train_anomalib.py), 无官方成品分发;
- SAM3 官方仓库(facebook/sam3)为 gated, 需在 Hugging Face 手动同意许可;
  本脚本走非 gated 的社区镜像, 无需 token。
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

# SAM3 权重来源: unsloth/sam3.1(非 gated 社区镜像, 与 facebook/sam3.1 同权重)
# 官方 facebook/sam3 与 facebook/sam3.1 均为 gated=manual, 需要网页上手动同意许可,
# 因此这里默认走免 token 的镜像; 已有官方权重者可跳过本步骤。
SAM3_REPO = "unsloth/sam3.1"
SAM3_REPO_TYPE = "model"
SAM3_BASE = f"https://huggingface.co/{SAM3_REPO}/resolve/main"
SAM3_DIR = WEIGHTS_DIR / "sam3"
SAM3_FILES = (
    "model.safetensors",        # 约 1.6GB fp16
    "config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
)

SAM3_GUIDE = f"""
SAM3(可选)权重获取指引
=========================
来源: {SAM3_REPO}(非 gated 社区镜像, 无需 token)
目标: {SAM3_DIR}
方式一(推荐, 自动):
    python scripts/download_assets.py --sam3
方式二(手动): 逐个下载以下文件到 {SAM3_DIR}
""" + "".join(f"    {SAM3_BASE}/{name}\n" for name in SAM3_FILES) + """
放好后重启后端, /api/models → sam3.available 会变为 true;
推理入口: backend/core/sam3_plugin.py, 接口 POST /api/segment。
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
    parser.add_argument("--sam3", action="store_true", help="自动下载 SAM3 权重(约 1.6GB)")
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

    if args.sam3:
        print(f"SAM3 权重下载自 {SAM3_REPO}(非 gated 镜像, 约 1.6GB)")
        for name in SAM3_FILES:
            target = SAM3_DIR / name
            if target.is_file() and target.stat().st_size > 0:
                print(f"[跳过] 已存在: {name} ({target.stat().st_size // 1024 // 1024}MB)")
                continue
            ok = download(f"{SAM3_BASE}/{name}", target) and ok
        print(f"\n完成。状态检查: GET /api/models → sam3.available")
    else:
        print(SAM3_GUIDE)
        print()

    if args.mvtec:
        print(MVTEC_GUIDE)
    elif not args.yolo and not args.sam3:
        print(MVTEC_GUIDE)

    if not args.yolo and not args.mvtec and not args.sam3:
        print("anomalib 瑕疵模型: 无官方成品, 请用 training/train_anomalib.py 自行训练。")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
