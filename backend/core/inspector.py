"""anomalib 瑕疵检测引擎封装.

职责:
- 扫描 weights/anomalib/<category>/*.ckpt, 形成"类别 → 模型"注册表;
- 懒加载模型并缓存, 推理输出: 异常分数 + 异常热力图;
- 生成 JET 伪彩热力图 PNG 与原图叠加 PNG(base64 编码)供前端直接渲染;
- 任意环节缺失(无 ckpt / anomalib 未安装)都抛出 ModelNotReadyError, 由 API 层转 503。

兼容性说明(已按 anomalib 2.7 实测校准):
- 首选 anomalib.deploy.TorchInferencer; 2.7 下直接吃 .ckpt 会抛错, 被接住后回退;
- 回退路径为 Engine.predict, 必须显式传 model 实例(否则 trainer 未赋值报错),
  且 Engine 需要 default_root_dir 指向 runs/ 下, 避免在项目根写 results/;
- 模型类由 ckpt 文件名前缀推断(efficientad_20261029_130635.ckpt → EfficientAd),
  推不出(如 last.ckpt)时依次尝试全部已知模型类。
"""

from __future__ import annotations

import base64
import io
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import config
from .detector import ModelNotReadyError

# 已知模型类名候选(兼容 anomalib 各版本命名大小写); 新增算法时在此登记
_KNOWN_MODEL_CLASS_NAMES: tuple[str, ...] = (
    "EfficientAd",
    "Efficientad",
    "Patchcore",
    "PatchCore",
)


# ---------------------------------------------------------------------------
# 工具: 伪彩与 PNG 编码(仅用 numpy + PIL, 避免强依赖 cv2)
# ---------------------------------------------------------------------------
def _to_numpy(value: Any) -> np.ndarray:
    """把 anomalib 返回的 tensor / ndarray 统一成 numpy。"""
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):  # torch.Tensor
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_heatmap(anomaly_map: np.ndarray) -> np.ndarray:
    """异常图 min-max 归一化到 [0, 1], 并压平到二维 (H, W)。"""
    heat = np.squeeze(np.asarray(anomaly_map, dtype=np.float32))
    if heat.ndim != 2:
        # 异常情况下取第一个通道, 保证后续流程不崩
        heat = heat.reshape(-1, heat.shape[-2], heat.shape[-1])[0] if heat.ndim >= 3 else heat.ravel()
        if heat.ndim != 2:
            size = int(np.sqrt(heat.size))
            heat = heat[: size * size].reshape(size, size)
    lo, hi = float(heat.min()), float(heat.max())
    if hi - lo < 1e-8:
        return np.zeros_like(heat, dtype=np.float32)
    return (heat - lo) / (hi - lo)


def _jet_colormap(norm: np.ndarray) -> np.ndarray:
    """纯 numpy 的 JET 伪彩: 输入 [0,1] 的 (H, W), 输出 uint8 的 (H, W, 3)。"""
    n = np.clip(norm, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * n - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * n - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * n - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _png_base64(rgb: np.ndarray) -> str:
    """(H, W, 3) 数组 → base64 PNG 字符串(防御性转 uint8)。"""
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _model_classes_for_ckpt(ckpt_name: str) -> list[type]:
    """由 ckpt 文件名前缀推断模型类, 按"推断结果优先 + 已知类兜底"排序。

    efficientad_20261029_130635.ckpt → [EfficientAd, 其余已知类...]
    last.ckpt(推不出前缀)           → 全部已知类
    """
    try:
        import anomalib.models as models_mod
    except ImportError as exc:
        raise ModelNotReadyError(
            "anomalib 未安装, 请在 .venv 中执行 pip install -r requirements.txt"
        ) from exc

    available: list[type] = []
    for name in _KNOWN_MODEL_CLASS_NAMES:
        cls = getattr(models_mod, name, None)
        if cls is not None and cls not in available:
            available.append(cls)

    prefix = Path(ckpt_name).stem.split("_")[0].lower()
    preferred = [c for c in available if c.__name__.lower().startswith(prefix)] if prefix else []
    ordered = preferred + [c for c in available if c not in preferred]
    if not ordered:
        raise ModelNotReadyError("当前 anomalib 版本中没有可用的已知模型类")
    return ordered


# ---------------------------------------------------------------------------
# 单个类别的模型句柄
# ---------------------------------------------------------------------------
class _CategoryModel:
    """一个类别的 anomalib 推理器(优先 TorchInferencer, 失败则走 Engine)。"""

    def __init__(self, category: str, ckpt_path: Path) -> None:
        self.category = category
        self.ckpt_path = ckpt_path
        self._inferencer: Any = None       # TorchInferencer 实例(首选)
        self._engine: Any = None           # anomalib Engine(备选)
        self._model_cls: type | None = None  # Engine 路径下实测可用的模型类
        self._mode: str | None = None      # "torch_inferencer" | "engine"

    def _ensure_loaded(self) -> None:
        if self._mode is not None:
            return

        # 路径 1: anomalib.deploy.TorchInferencer(支持直接吃 .ckpt 的版本)
        try:
            from anomalib.deploy import TorchInferencer

            self._inferencer = TorchInferencer(path=str(self.ckpt_path))
            self._mode = "torch_inferencer"
            return
        except Exception:
            # 该版本 TorchInferencer 不支持直接加载 ckpt(anomalib 2.7 实测如此),
            # 继续尝试 Engine 路径
            self._inferencer = None

        # 路径 2: anomalib Engine.predict
        # 注意: 必须传 default_root_dir, 否则会在项目根目录写 results/;
        # 且 predict 必须显式传 model 实例(见 predict())。
        try:
            from anomalib.engine import Engine

            runtime_dir = Path(config.RUNS_DIR / "inspect_runtime")
            runtime_dir.mkdir(parents=True, exist_ok=True)
            # 关闭进度条/日志器: 服务控制台多为 GBK, lightning 进度条的
            # Unicode 字符会触发 UnicodeEncodeError(实测 2.7 踩中)
            engine_kwargs: dict[str, Any] = {
                "default_root_dir": str(runtime_dir),
                "enable_progress_bar": False,
                "logger": False,
            }
            try:
                self._engine = Engine(**engine_kwargs)
            except TypeError:
                # 旧版 anomalib 不透传这些 Trainer 参数
                self._engine = Engine(default_root_dir=str(runtime_dir))
            self._mode = "engine"
        except Exception as exc:
            raise ModelNotReadyError(
                f"anomalib 不可用或无法加载 {self.ckpt_path.name}: {exc}"
            ) from exc

    def _predict_with_engine(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        """Engine 路径: PredictDataset 需要文件路径, 写临时 PNG。

        模型类按 ckpt 文件名前缀推断并缓存; 推断不准时(如 last.ckpt)
        依次尝试其余已知模型类, 第一个跑通的即缓存复用。
        """
        from anomalib.data import PredictDataset

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            Image.fromarray(image).save(tmp_path)
            dataset = PredictDataset(path=str(tmp_path))
            candidates = (
                [self._model_cls]
                if self._model_cls is not None
                else _model_classes_for_ckpt(self.ckpt_path.name)
            )
            last_error: Exception | None = None
            for cls in candidates:
                try:
                    predictions = self._engine.predict(
                        model=cls(), dataset=dataset, ckpt_path=str(self.ckpt_path)
                    )
                    self._model_cls = cls  # 缓存实测可用的类
                    break
                except Exception as exc:  # noqa: BLE001 - 换下一个候选类继续试
                    last_error = exc
            else:
                raise ModelNotReadyError(
                    f"无法用已知模型类加载 {self.ckpt_path.name}: {last_error}"
                ) from last_error
        finally:
            tmp_path.unlink(missing_ok=True)

        preds = list(predictions or [])
        if not preds:
            raise RuntimeError("anomalib 未返回任何预测结果")
        pred = preds[0]
        score = float(_to_numpy(pred.pred_score).reshape(-1)[0])
        anomaly_map = _to_numpy(pred.anomaly_map)
        return score, anomaly_map

    def predict(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        """返回 (图像级异常分数, 像素级异常图 HxW)。"""
        self._ensure_loaded()

        if self._mode == "torch_inferencer":
            result = self._inferencer.predict(image)
            score = float(_to_numpy(result.pred_score).reshape(-1)[0])
            anomaly_map = _to_numpy(result.anomaly_map)
            return score, anomaly_map

        return self._predict_with_engine(image)


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
class AnomalyInspector:
    """瑕疵检测模型注册表 + 推理入口。"""

    def __init__(self, weights_dir: Path | None = None) -> None:
        self.weights_dir = weights_dir or config.ANOMALIB_WEIGHTS_DIR
        self._models: dict[str, _CategoryModel] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 发现与状态
    # ------------------------------------------------------------------
    @staticmethod
    def _is_named_ckpt(path: Path) -> bool:
        """文件名是否带已知模型前缀(如 efficientad_20261029_130635.ckpt)。"""
        prefix = path.stem.split("_")[0].lower()
        return any(name.lower().startswith(prefix) for name in _KNOWN_MODEL_CLASS_NAMES) and prefix != "last"

    def _find_ckpt(self, category: str) -> Path | None:
        """选定该类别的推理权重。

        优先带模型名前缀的时间戳文件(可据此推断模型类), 取最新;
        没有时兜底用最新 .ckpt(如 last.ckpt, 模型类逐个试)。
        """
        cat_dir = self.weights_dir / category
        if not cat_dir.is_dir():
            return None
        ckpts = sorted(cat_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        if not ckpts:
            return None
        named = [p for p in ckpts if self._is_named_ckpt(p)]
        return (named or ckpts)[-1]

    def available_categories(self) -> list[dict]:
        """扫描权重目录, 返回每个已注册类别的信息。"""
        categories: list[dict] = []
        if not self.weights_dir.is_dir():
            return categories
        for cat_dir in sorted(self.weights_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            if not any(cat_dir.glob("*.ckpt")):
                continue
            selected = self._find_ckpt(cat_dir.name)
            categories.append(
                {
                    "name": cat_dir.name,
                    "ckpt": selected.name if selected else None,
                    "loaded": cat_dir.name in self._models,
                    "threshold": self._threshold(cat_dir.name),
                }
            )
        return categories

    def _threshold(self, category: str) -> float:
        """读取类别专属阈值 threshold.json, 缺省用全局默认值。"""
        meta = self.weights_dir / category / "threshold.json"
        if meta.is_file():
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                return float(data.get("threshold", config.DEFAULT_ANOMALY_THRESHOLD))
            except Exception:  # noqa: BLE001 - 配置文件损坏时回落默认值
                pass
        return config.DEFAULT_ANOMALY_THRESHOLD

    def status(self) -> dict:
        categories = self.available_categories()
        return {
            "available": len(categories) > 0,
            "weights_dir": str(self.weights_dir),
            "categories": categories,
            "message": (
                f"已注册 {len(categories)} 个类别模型"
                if categories
                else "模型未就绪: weights/anomalib/<类别>/ 下没有 .ckpt, 请先用 training/train_anomalib.py 训练"
            ),
        }

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def _get_model(self, category: str) -> _CategoryModel:
        with self._lock:
            if category in self._models:
                return self._models[category]
            ckpt = self._find_ckpt(category)
            if ckpt is None:
                raise ModelNotReadyError(
                    f"类别 '{category}' 的模型未就绪: 未找到 {self.weights_dir / category}/*.ckpt"
                )
            model = _CategoryModel(category, ckpt)
            self._models[category] = model
            return model

    def inspect(
        self,
        category: str,
        image: np.ndarray,
        threshold: float | None = None,
    ) -> dict:
        """对一张 RGB 图像做瑕疵检测。

        返回 dict: score / is_ng / threshold / heatmap_png_base64 /
        overlay_png_base64 / inference_ms
        """
        model = self._get_model(category)
        thres = self._threshold(category) if threshold is None else float(threshold)

        start = time.perf_counter()
        score, anomaly_map = model.predict(image)
        inference_ms = (time.perf_counter() - start) * 1000.0

        # 热力图: 归一化 → JET 伪彩 → 缩放到原图尺寸
        heat01 = _normalize_heatmap(anomaly_map)
        h, w = image.shape[:2]
        heat_img = Image.fromarray(_jet_colormap(heat01)).resize((w, h), Image.BILINEAR)
        heat_rgb = np.asarray(heat_img, dtype=np.uint8)

        # 叠加: 原图 60% + 热力图 40%
        overlay = (0.6 * image.astype(np.float32) + 0.4 * heat_rgb.astype(np.float32))
        overlay_rgb = np.clip(overlay, 0, 255).astype(np.uint8)

        return {
            "score": round(score, 4),
            "is_ng": bool(score >= thres),
            "threshold": round(thres, 4),
            "heatmap_png_base64": _png_base64(heat_rgb),
            "overlay_png_base64": _png_base64(overlay_rgb),
            "inference_ms": round(inference_ms, 1),
        }


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------
_inspector: AnomalyInspector | None = None


def get_inspector() -> AnomalyInspector:
    global _inspector
    if _inspector is None:
        _inspector = AnomalyInspector()
    return _inspector
