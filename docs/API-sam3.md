# SAM3 分割接口契约 (v1, 冻结)

> 前后端唯一事实源。后端实现与前端对接都必须严格遵循本文件。

## 1. 模型与环境事实（实测）
- 权重: `unsloth/sam3.1` 的 `model.safetensors` (fp16, 1603 MB) → 本地 `weights/sam3/`
- 推理库: `transformers 5.21.0`（已装入 .venv；`Sam3Model` + `Sam3Processor` 导入正常）
- torch 2.5.1+cu121 / CUDA True / RTX 3060 Ti 8GB
- 官方用法（源码 docstring 实测）:
  ```python
  inputs = processor(images=image, text="bottle", return_tensors="pt")
  outputs = model(**inputs)
  res = processor.post_process_instance_segmentation(outputs, threshold=0.3, target_sizes=[image.size[::-1]])
  # res[0] -> {"scores": Tensor[N], "boxes": Tensor[N,4] xyxy, "masks": Tensor[N,H,W] bool}
  ```
- 权重缺失时**不得**抛未捕获异常，必须返回 MODEL_NOT_READY(503)。

## 2. GET /api/models（在既有响应中扩展 sam3 字段，其余字段保持不变）
```json
{
  "ok": true,
  "models": {
    "yolo": { "available": true, "message": "..." },
    "anomalib": { "available": true, "categories": [ ... ] },
    "sam3": {
      "available": true,
      "message": "sam3.1 fp16 就绪 (1.6GB)",
      "model": "sam3.1-fp16",
      "weights_mb": 1603,
      "concepts_hint": ["bottle", "screw", "scratch", "broken", "dent"]
    }
  }
}
```
- `available=false` 时 `message` 必须说明原因（未找到 weights/sam3/*.safetensors 等）。前端徽章直接显示该 message。

## 3. POST /api/segment（新增）
请求: `multipart/form-data`
| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| file | 文件 | 是 | jpg/png/bmp/webp |
| text | str | 是 | 概念提示，逗号分隔，如 `bottle,broken`；也接受单概念 |
| threshold | float | 否 | 实例分数阈值，默认 0.30，范围 0.05–0.95 |
| mask_threshold | float | 否 | mask 二值化阈值，默认 0.50 |

响应 200:
```json
{
  "ok": true,
  "result": {
    "concepts": ["bottle", "broken"],
    "model": "sam3.1-fp16",
    "device": "cuda",
    "inference_ms": 412.5,
    "image_size": [1024, 1024],
    "instances": [
      { "concept": "bottle", "score": 0.931, "box": [12.0, 30.0, 980.0, 1010.0],
        "area_px": 512000, "area_ratio": 0.49,
        "mask_png_base64": "<单实例黑白mask PNG>" }
    ],
    "mask_png_base64": "<所有实例的彩色叠加图 PNG, 每实例一色>",
    "overlay_png_base64": "<原图 + 半透明彩色mask + 框 PNG>"
  }
}
```
错误:
- `503` `{"ok":false,"error":{"code":"MODEL_NOT_READY","message":"..."}}` 权重缺失
- `400` `{"ok":false,"error":{"code":"VALIDATION_ERROR","message":"..."}}` 缺 text / 非法图片 / 阈值越界
- `500` `{"ok":false,"error":{"code":"INTERNAL_ERROR","message":"..."}}`

## 4. 前端约定
- 模式单选新增第三个值 `segment`（沿用 `input[name="mode"]`，现有 detect / inspect 行为不得改变）
- 新增 DOM（id 固定，不得改名）：`promptRow`, `promptInput`, `segThresholdRow`, `segThresholdSlider`, `segThresholdVal`, `maskRow`
- `input[name="view"]` 在 segment 模式下提供 `mask` 与 `overlay` 两个值；inspect 模式仍为 overlay/heatmap/original
- 结果用 `#verdict` 显示 `N 个实例`（kind=`info`），`#metaBox` 显示 `概念: x,y · 分数 max · 耗时 z ms`
- 仍必须零构建、零 CDN、可离线（不得引入外链字体/脚本）

## 5. 兼容约束（硬）
- 现有接口 `/api/health` `/api/detect` `/api/inspect` `/api/train*` 的请求/响应**不得变更**
- 现有 DOM id（dropzone, fileInput, dzFile, categoryRow, categorySelect, confRow, confSlider, confVal, threshRow, threshSlider, threshVal, runBtn, statusBar, verdict, metaBox, viewRow, stage, emptyHint, historyList, clearHistory, badgeYolo, badgeAnomalib, badgeSam3）**全部保留**
- 错误体统一 `{"ok":false,"error":{"code","message"}}`
