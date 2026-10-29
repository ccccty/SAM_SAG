"""数据集目录校验/整理工具(MVTec AD 结构)。

期望结构:
    data/mvtec/<category>/
    ├── train/good/            (正常训练图, 必需, 至少 1 张)
    ├── test/good/             (正常测试图, 建议)
    ├── test/<defect_type>/    (缺陷测试图, 建议, 可多个)
    └── ground_truth/<defect>/ (缺陷掩码, 可选)

用法:
    python training/prepare_dataset.py                     # 校验所有类别
    python training/prepare_dataset.py --category bottle   # 校验单个类别
    python training/prepare_dataset.py --category bottle --fix   # 补齐缺失目录
退出码: 0 = 全部通过; 1 = 存在问题。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MVTEC_ROOT = BASE_DIR / "data" / "mvtec"
IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.iterdir() if p.suffix.lower() in IMG_SUFFIXES)


def check_category(cat_dir: Path, fix: bool) -> list[str]:
    """校验单个类别, 返回问题列表(空 = 通过)。"""
    problems: list[str] = []
    train_good = cat_dir / "train" / "good"
    test_dir = cat_dir / "test"

    if fix:
        train_good.mkdir(parents=True, exist_ok=True)
        (test_dir / "good").mkdir(parents=True, exist_ok=True)

    if not train_good.is_dir():
        problems.append("缺少 train/good/ 目录(正常训练图)")
    elif count_images(train_good) == 0:
        problems.append("train/good/ 下没有任何图片")

    if not test_dir.is_dir():
        problems.append("缺少 test/ 目录(评估与阈值估计需要)")
    else:
        subdirs = [d for d in test_dir.iterdir() if d.is_dir()]
        if not subdirs:
            problems.append("test/ 下没有子目录(至少应有 good/ 与若干缺陷目录)")
        defect_dirs = [d for d in subdirs if d.name != "good"]
        if not defect_dirs:
            problems.append("test/ 下没有缺陷目录(如 broken_large/), 无法评估检测效果")
    return problems


def report(cat_dir: Path, problems: list[str]) -> None:
    name = cat_dir.name
    n_train = count_images(cat_dir / "train" / "good")
    test_dir = cat_dir / "test"
    test_info = []
    if test_dir.is_dir():
        for d in sorted(test_dir.iterdir()):
            if d.is_dir():
                test_info.append(f"{d.name}({count_images(d)})")
    print(f"  类别 {name}: train/good={n_train} 张, test: {', '.join(test_info) or '空'}")
    if problems:
        for p in problems:
            print(f"    ✗ {p}")
    else:
        print("    ✓ 结构完整")


def main() -> int:
    parser = argparse.ArgumentParser(description="校验/整理 MVTec AD 数据集目录结构")
    parser.add_argument("--category", default=None, help="只处理指定类别(默认全部)")
    parser.add_argument("--fix", action="store_true", help="创建缺失的标准目录")
    args = parser.parse_args()

    if args.category:
        targets = [MVTEC_ROOT / args.category]
        if args.fix:
            targets[0].mkdir(parents=True, exist_ok=True)
    else:
        if not MVTEC_ROOT.is_dir():
            print(f"[提示] {MVTEC_ROOT} 不存在。")
            if args.fix:
                MVTEC_ROOT.mkdir(parents=True)
                print(f"       已创建 {MVTEC_ROOT}, 请按 README 放入各类别数据。")
                return 0
            print("       可用 --fix 创建骨架, 或参考 scripts/download_assets.py --mvtec 的下载指引。")
            return 1
        targets = [d for d in sorted(MVTEC_ROOT.iterdir()) if d.is_dir()]

    if not targets:
        print(f"[提示] {MVTEC_ROOT} 下没有任何类别目录。")
        return 1

    all_ok = True
    print(f"[SAM_SAG] 数据集根目录: {MVTEC_ROOT}")
    for cat_dir in targets:
        if not cat_dir.is_dir():
            print(f"  类别 {cat_dir.name}: ✗ 目录不存在")
            all_ok = False
            continue
        problems = check_category(cat_dir, args.fix)
        report(cat_dir, problems)
        all_ok = all_ok and not problems

    print()
    print("[结果] 全部通过 ✓" if all_ok else "[结果] 存在上述问题, 修复后再训练 ✗")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
