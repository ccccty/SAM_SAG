"""Environment verification for SAM_SAG visual-inspection stack."""
import importlib
import importlib.metadata
import sys


def pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> int:
    print(f"python: {sys.version.split()[0]} ({sys.executable})")

    import torch
    print(f"torch: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")
        x = torch.rand(4, 4, device="cuda")
        print(f"cuda tensor ok: {x.is_cuda}, sum={x.sum().item():.4f}")

    print(f"torchvision: {pkg_version('torchvision')}")
    print(f"ultralytics: {pkg_version('ultralytics')}")
    print(f"fastapi: {pkg_version('fastapi')}")
    print(f"uvicorn: {pkg_version('uvicorn')}")
    print(f"pillow: {pkg_version('pillow')}")
    print(f"opencv-python-headless: {pkg_version('opencv-python-headless')}")
    print(f"python-multipart: {pkg_version('python-multipart')}")
    print(f"anomalib: {pkg_version('anomalib')}")

    try:
        from ultralytics import YOLO  # noqa: F401
        print("ultralytics import: OK")
    except Exception as exc:  # noqa: BLE001
        print(f"ultralytics import: FAIL ({exc})")
        return 1

    try:
        from anomalib.models import Patchcore  # noqa: F401
        print("anomalib.models.Patchcore import: OK")
    except Exception as exc:  # noqa: BLE001
        print(f"anomalib Patchcore import: FAIL ({exc})")
        try:
            from anomalib.models import EfficientAd  # noqa: F401
            print("anomalib.models.EfficientAd import: OK")
        except Exception as exc2:  # noqa: BLE001
            print(f"anomalib EfficientAd import: FAIL ({exc2})")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
