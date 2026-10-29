from pathlib import Path
from PIL import Image
from ultralytics import YOLO

root = Path(r"D:\sam\SAM_SAG")
m = YOLO(str(root / "weights" / "yolo11n.pt"))
print("yolo11n.pt loaded, task:", m.task, "classes:", len(m.names))

mv = root / "data" / "mvtec"
for cat in ("bottle", "screw"):
    cdir = mv / cat
    parts = {}
    nfiles = 0
    nbytes = 0
    for p in cdir.rglob("*"):
        if p.is_file():
            nfiles += 1
            nbytes += p.stat().st_size
            key = str(p.relative_to(cdir).parent)
            parts[key] = parts.get(key, 0) + 1
    print(f"[{cat}] files={nfiles} total={nbytes/1e6:.1f} MB")
    for k in sorted(parts):
        print("   ", k, parts[k])
    sample = next((cdir / "train" / "good").glob("*.png"))
    with Image.open(sample) as im:
        im.verify()
    with Image.open(sample) as im2:
        print(f"    sample {sample.name}: format={im2.format} size={im2.size}")