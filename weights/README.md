# 权重放置说明

本目录用于存放模型权重文件。**权重文件不纳入 Git 管理**(.gitignore 已排除),请按下表放置:

| 文件/目录 | 用途 | 获取方式 |
| --- | --- | --- |
| `yolo11n.pt` | YOLO 目标检测权重(`/api/detect` 使用) | `python scripts/download_assets.py --yolo` 自动下载,或训练后由 `training/train_yolo.py` 自动复制 |
| `anomalib/<类别>/*.ckpt` | anomalib 瑕疵检测模型,按类别分目录(`/api/inspect` 使用) | `python training/train_anomalib.py --category <类别>` 训练生成 |
| `anomalib/<类别>/threshold.json` | (可选)类别判定阈值,如 `{"threshold": 0.5}` | 训练脚本自动写入,可手工调整 |
| `sam3/` | (可选)SAM3 权重目录,放入后 SAM3 插件自动标记 available | 待 SAM3 官方发布后按其说明放置 |

缺失任一权重只会使对应接口返回 `503 模型未就绪`,不影响其他功能。
