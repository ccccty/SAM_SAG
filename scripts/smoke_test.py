"""SAM_SAG 冒烟测试: 对运行中的后端打一通核心接口。

前置: 后端已启动(start.bat 或 python -m backend.main)。
用法:
    python scripts/smoke_test.py [--base http://127.0.0.1:8000] [--image path/to/img.png]

覆盖:
    1. GET  /api/health        必须 200 且 ok
    2. GET  /api/models        必须 200; 读取各引擎可用性
    3. POST /api/detect        YOLO 可用时必须成功, 否则必须返回 503+MODEL_NOT_READY
    4. POST /api/inspect       存在已注册类别时必须成功, 否则必须返回 503/400 错误契约
退出码 0 = 全部符合预期。
"""

from __future__ import annotations

import argparse
import base64
import io
import sys

# 1x1 红色 PNG, 作为无测试图时的兜底输入
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def make_test_image() -> bytes:
    """优先用 Pillow 生成 640x480 带方块的测试图; 没有 Pillow 就用内置 1x1 PNG。"""
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (640, 480), (40, 44, 52))
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 300, 300], fill=(200, 60, 60))
        draw.ellipse([350, 150, 550, 350], fill=(60, 160, 220))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        return base64.b64decode(_TINY_PNG_B64)


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM_SAG 后端冒烟测试")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="后端地址")
    parser.add_argument("--image", default=None, help="测试图片路径(缺省自动生成)")
    parser.add_argument("--timeout", type=float, default=120.0, help="单次请求超时(秒)")
    args = parser.parse_args()

    try:
        import requests
    except ImportError:
        print("[FAIL] requests 未安装: pip install requests", file=sys.stderr)
        return 1

    base = args.base.rstrip("/")
    results: list[bool] = []

    # 1. health
    try:
        r = requests.get(base + "/api/health", timeout=args.timeout)
        j = r.json()
        results.append(check("GET /api/health", r.status_code == 200 and j.get("ok") is True,
                             f"service={j.get('service')} cuda={j.get('cuda_available')}"))
    except Exception as exc:
        results.append(check("GET /api/health", False, f"连接失败: {exc}(后端是否已启动?)"))
        print("\n后端不可达, 后续用例跳过。")
        return 1

    # 2. models
    models = {}
    try:
        r = requests.get(base + "/api/models", timeout=args.timeout)
        j = r.json()
        models = j.get("models", {})
        ok = r.status_code == 200 and j.get("ok") is True and "yolo" in models and "anomalib" in models
        results.append(check("GET /api/models", ok,
                             "yolo=" + str(models.get("yolo", {}).get("available"))
                             + " anomalib_categories="
                             + str(len(models.get("anomalib", {}).get("categories", [])))))
    except Exception as exc:
        results.append(check("GET /api/models", False, str(exc)))

    # 测试图
    if args.image:
        img_bytes = open(args.image, "rb").read()
    else:
        img_bytes = make_test_image()

    # 3. detect
    try:
        r = requests.post(base + "/api/detect", files={"file": ("smoke.png", img_bytes, "image/png")},
                          timeout=args.timeout)
        j = r.json()
        if r.status_code == 200 and j.get("ok"):
            results.append(check("POST /api/detect", True,
                                 f"detections={len(j.get('detections', []))} ms={j.get('inference_ms')}"))
        elif r.status_code == 503 and j.get("error", {}).get("code") == "MODEL_NOT_READY":
            expected = not models.get("yolo", {}).get("available", False)
            results.append(check("POST /api/detect(模型未就绪分支)", expected,
                                 "返回 503 且与 /api/models 状态一致" if expected
                                 else "/api/models 报告可用但 /api/detect 报未就绪, 状态不一致"))
        else:
            results.append(check("POST /api/detect", False, f"status={r.status_code} body={j}"))
    except Exception as exc:
        results.append(check("POST /api/detect", False, str(exc)))

    # 4. inspect
    categories = models.get("anomalib", {}).get("categories", [])
    try:
        if categories:
            cat = categories[0]["name"]
            r = requests.post(base + "/api/inspect",
                              files={"file": ("smoke.png", img_bytes, "image/png")},
                              data={"category": cat}, timeout=args.timeout)
            j = r.json()
            ok = (r.status_code == 200 and j.get("ok")
                  and "score" in j and "is_ng" in j and "overlay_png_base64" in j)
            results.append(check("POST /api/inspect", ok,
                                 f"category={cat} score={j.get('score')} is_ng={j.get('is_ng')}"))
        else:
            r = requests.post(base + "/api/inspect",
                              files={"file": ("smoke.png", img_bytes, "image/png")},
                              data={"category": "__nonexistent__"}, timeout=args.timeout)
            j = r.json()
            ok = r.status_code in (400, 503) and j.get("ok") is False and "error" in j
            results.append(check("POST /api/inspect(无类别模型分支)", ok,
                                 f"status={r.status_code} code={j.get('error', {}).get('code')}"))
    except Exception as exc:
        results.append(check("POST /api/inspect", False, str(exc)))

    passed = sum(1 for x in results if x)
    print(f"\n冒烟结果: {passed}/{len(results)} 通过")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
