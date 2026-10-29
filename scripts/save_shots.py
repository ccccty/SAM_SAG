import json, base64, sys, pathlib
resp = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
name = sys.argv[2]
out = pathlib.Path("docs/screenshots"); out.mkdir(parents=True, exist_ok=True)
assert resp.get("ok"), resp
for key in ("heatmap_png_base64", "overlay_png_base64"):
    (out / f"{name}_{key.split('_')[0]}.png").write_bytes(base64.b64decode(resp[key]))
print(json.dumps({k: resp[k] for k in ("category","score","is_ng","threshold","inference_ms")}, ensure_ascii=False))
