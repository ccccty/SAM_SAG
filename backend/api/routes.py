"""SAM_SAG HTTP API 路由.

接口契约(与 frontend/app.js 严格一致):
- GET  /api/health            → {ok, service, version, time, cuda_available}
- GET  /api/models            → {ok, models: {yolo, anomalib, sam3}}
- POST /api/detect            → {ok, detections: [{label, confidence, box}], inference_ms}
- POST /api/inspect           → {ok, category, score, is_ng, threshold,
                                 heatmap_png_base64, overlay_png_base64, inference_ms}
- POST /api/train             → {ok, task_id}            启动后台训练
- GET  /api/train/{task_id}   → {ok, task: {...}}        查询训练状态
- GET  /api/train             → {ok, tasks: [...]}       训练任务列表
错误统一为 {ok: false, error: {code, message}} 并带合适的 HTTP 状态码。
"""

from __future__ import annotations

import io
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from ..core import config
from ..core.detector import ModelNotReadyError, get_detector
from ..core.inspector import get_inspector
from ..core.sam3_plugin import get_sam3_plugin

router = APIRouter()


# ---------------------------------------------------------------------------
# 统一错误
# ---------------------------------------------------------------------------
class ApiError(Exception):
    """业务错误: 由 main.py 的异常处理器转成统一 JSON。"""

    def __init__(self, http_status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.code = code
        self.message = message


def _model_not_ready(exc: ModelNotReadyError) -> ApiError:
    return ApiError(503, "MODEL_NOT_READY", f"模型未就绪: {exc}")


# ---------------------------------------------------------------------------
# 上传图片解码
# ---------------------------------------------------------------------------
async def _read_image(file: UploadFile) -> np.ndarray:
    """读取上传文件 → RGB uint8 数组; 非法输入抛 ApiError(400)。"""
    content = await file.read()
    if not content:
        raise ApiError(400, "EMPTY_FILE", "上传的文件为空")
    if len(content) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise ApiError(413, "FILE_TOO_LARGE", f"图片超过 {config.MAX_UPLOAD_MB}MB 上限")
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise ApiError(400, "INVALID_IMAGE", f"无法解析图片文件: {exc}") from exc
    return np.asarray(image, dtype=np.uint8)


# ---------------------------------------------------------------------------
# 健康检查与模型状态
# ---------------------------------------------------------------------------
@router.get("/health")
def health() -> dict:
    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - torch 未安装也算健康, 只是没 GPU
        pass
    return {
        "ok": True,
        "service": "SAM_SAG",
        "version": _app_version(),
        "time": datetime.now(timezone.utc).isoformat(),
        "cuda_available": cuda_available,
    }


def _app_version() -> str:
    from .. import __version__

    return __version__


@router.get("/models")
def models() -> dict:
    """汇总三个引擎的可用状态, 前端据此点亮/置灰对应功能。"""
    return {
        "ok": True,
        "models": {
            "yolo": get_detector().status(),
            "anomalib": get_inspector().status(),
            "sam3": get_sam3_plugin().status(),
        },
    }


# ---------------------------------------------------------------------------
# 目标检测
# ---------------------------------------------------------------------------
@router.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Form(config.DEFAULT_CONF_THRESHOLD),
) -> dict:
    image = await _read_image(file)
    try:
        detections, inference_ms = get_detector().detect(image, conf=conf)
    except ModelNotReadyError as exc:
        raise _model_not_ready(exc) from exc
    return {
        "ok": True,
        "detections": detections,
        "inference_ms": round(inference_ms, 1),
    }


# ---------------------------------------------------------------------------
# 瑕疵检测
# ---------------------------------------------------------------------------
@router.post("/inspect")
async def inspect(
    file: UploadFile = File(...),
    category: str = Form(...),
    threshold: float | None = Form(None),
) -> dict:
    image = await _read_image(file)
    category = category.strip()
    if not category:
        raise ApiError(400, "MISSING_CATEGORY", "缺少 category 参数(瑕疵模型类别)")
    try:
        result = get_inspector().inspect(category, image, threshold=threshold)
    except ModelNotReadyError as exc:
        raise _model_not_ready(exc) from exc
    return {"ok": True, "category": category, **result}


# ---------------------------------------------------------------------------
# 后台训练任务
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    """训练任务参数。kind: "yolo" | "anomalib"。"""

    kind: str = Field(..., pattern="^(yolo|anomalib)$")
    category: str | None = None          # anomalib: MVTec 类别名
    model: str = "efficientad"           # anomalib: efficientad | patchcore
    data_yaml: str | None = None         # yolo: 数据集 yaml 路径
    epochs: int = 50
    batch_size: int | None = None        # None → 用脚本内默认值(适配 8GB 显存)
    image_size: int | None = None        # None → 脚本默认(256 / 640)


class _TrainTask:
    """一个后台训练进程的运行记录。"""

    def __init__(self, task_id: str, kind: str, command: list[str], log_path: Path) -> None:
        self.id = task_id
        self.kind = kind
        self.command = command
        self.log_path = log_path
        self.status = "pending"  # pending → running → succeeded / failed
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.exit_code: int | None = None
        self.error: str | None = None

    def to_dict(self) -> dict:
        log_tail = ""
        if self.log_path.is_file():
            try:
                text = self.log_path.read_text(encoding="utf-8", errors="replace")
                log_tail = text[-config.TRAIN_LOG_TAIL_CHARS :]
            except OSError:
                pass
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "command": self.command,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "error": self.error,
            "log_tail": log_tail,
        }


_train_tasks: dict[str, _TrainTask] = {}
_train_lock = threading.Lock()
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_-]+$")


def _run_train_task(task: _TrainTask) -> None:
    """线程体: 拉起训练子进程并把输出写入日志文件。"""
    task.status = "running"
    task.started_at = datetime.now(timezone.utc).isoformat()
    config.TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with task.log_path.open("w", encoding="utf-8", errors="replace") as log:
            log.write("$ " + " ".join(task.command) + "\n\n")
            log.flush()
            proc = subprocess.Popen(
                task.command,
                cwd=str(config.BASE_DIR),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            task.exit_code = proc.wait()
        task.status = "succeeded" if task.exit_code == 0 else "failed"
    except Exception as exc:  # noqa: BLE001 - 任何拉起失败都记入任务
        task.status = "failed"
        task.error = str(exc)
    finally:
        task.finished_at = datetime.now(timezone.utc).isoformat()


def _build_train_command(req: TrainRequest) -> list[str]:
    """把训练请求翻译成对 training/ 脚本的命令行(不经过 shell, 无注入面)。"""
    py = sys.executable
    if req.kind == "anomalib":
        if not req.category or not _SAFE_TOKEN.match(req.category):
            raise ApiError(400, "INVALID_CATEGORY", "anomalib 训练需要合法的 category(字母/数字/下划线/连字符)")
        cmd = [
            py, str(config.BASE_DIR / "training" / "train_anomalib.py"),
            "--category", req.category,
            "--model", req.model,
            "--epochs", str(req.epochs),
        ]
        if req.batch_size:
            cmd += ["--batch-size", str(req.batch_size)]
        if req.image_size:
            cmd += ["--image-size", str(req.image_size)]
        return cmd

    # yolo
    if not req.data_yaml:
        raise ApiError(400, "MISSING_DATA_YAML", "YOLO 训练需要 data_yaml(数据集配置)路径")
    data_yaml = Path(req.data_yaml)
    if not data_yaml.is_file():
        raise ApiError(400, "DATA_YAML_NOT_FOUND", f"数据集配置不存在: {req.data_yaml}")
    cmd = [
        py, str(config.BASE_DIR / "training" / "train_yolo.py"),
        "--data", str(data_yaml),
        "--epochs", str(req.epochs),
    ]
    if req.batch_size:
        cmd += ["--batch", str(req.batch_size)]
    if req.image_size:
        cmd += ["--imgsz", str(req.image_size)]
    return cmd


@router.post("/train")
def start_train(req: TrainRequest) -> dict:
    command = _build_train_command(req)
    task_id = uuid.uuid4().hex[:12]
    log_path = config.TRAIN_LOG_DIR / f"train_{task_id}.log"
    task = _TrainTask(task_id, req.kind, command, log_path)
    with _train_lock:
        _train_tasks[task_id] = task
    threading.Thread(target=_run_train_task, args=(task,), daemon=True).start()
    return {"ok": True, "task_id": task_id}


@router.get("/train")
def list_trains() -> dict:
    with _train_lock:
        tasks = [t.to_dict() for t in _train_tasks.values()]
    tasks.sort(key=lambda t: t["created_at"], reverse=True)
    return {"ok": True, "tasks": tasks}


@router.get("/train/{task_id}")
def get_train(task_id: str) -> dict:
    with _train_lock:
        task = _train_tasks.get(task_id)
    if task is None:
        raise ApiError(404, "TASK_NOT_FOUND", f"训练任务不存在: {task_id}")
    return {"ok": True, "task": task.to_dict()}
