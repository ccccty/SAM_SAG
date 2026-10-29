"""SAM_SAG 全局配置.

所有路径、阈值、服务参数集中在此, 便于与训练脚本、启动脚本保持一致。
路径约定(与 README / training 脚本对齐):
    weights/yolo11n.pt                     YOLO 检测权重
    weights/anomalib/<category>/*.ckpt     anomalib 瑕疵检测模型(按类别)
    weights/sam3/                          SAM3 可选插件权重目录
    data/mvtec/<category>/{train,test}/    MVTec AD 数据集
    runs/                                  训练产物与训练日志
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
# 本文件位于 backend/core/config.py, 向上三级即项目根目录。
BASE_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"

WEIGHTS_DIR = BASE_DIR / "weights"
YOLO_WEIGHTS = WEIGHTS_DIR / "yolo11n.pt"
ANOMALIB_WEIGHTS_DIR = WEIGHTS_DIR / "anomalib"
SAM3_WEIGHTS_DIR = WEIGHTS_DIR / "sam3"

DATA_DIR = BASE_DIR / "data"
MVTEC_DIR = DATA_DIR / "mvtec"

RUNS_DIR = BASE_DIR / "runs"
TRAIN_LOG_DIR = RUNS_DIR / "train_logs"

# ---------------------------------------------------------------------------
# 服务
# ---------------------------------------------------------------------------
HOST = os.environ.get("SAM_SAG_HOST", "127.0.0.1")
PORT = int(os.environ.get("SAM_SAG_PORT", "8000"))
APP_TITLE = "SAM_SAG 视觉质检上位机"

# ---------------------------------------------------------------------------
# 推理默认值
# ---------------------------------------------------------------------------
DEFAULT_CONF_THRESHOLD = 0.25      # YOLO 置信度阈值
DEFAULT_ANOMALY_THRESHOLD = 0.5    # anomalib 异常分数阈值(0~1, 越大越严格)
SAM3_DEFAULT_THRESHOLD = 0.30      # SAM3 实例分数阈值(契约范围 0.05~0.95)
SAM3_DEFAULT_MASK_THRESHOLD = 0.50 # SAM3 mask 二值化阈值
MAX_UPLOAD_MB = 20                 # 上传图片大小上限
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# 训练任务
# ---------------------------------------------------------------------------
TRAIN_LOG_TAIL_CHARS = 4000        # 状态接口返回的日志尾部最大字符数


def ensure_dirs() -> None:
    """启动时确保运行期目录存在(weights/data 由用户/训练脚本填充)."""
    for d in (WEIGHTS_DIR, ANOMALIB_WEIGHTS_DIR, RUNS_DIR, TRAIN_LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
