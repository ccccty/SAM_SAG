"""Download selected MVTec AD categories from a public HuggingFace mirror."""
import concurrent.futures as cf
import json
import sys
import urllib.request
from pathlib import Path

REPO = "foersben/mvtec-ad"
BASE = "https://huggingface.co"
OUT = Path(r"D:\sam\SAM_SAG\data\mvtec")
CATEGORIES = ["bottle", "screw"]
TIMEOUT = 120

def get(url, dest=None, retries=3):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sam-sag-setup/1.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                data = r.read()
            if dest is not None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                return dest.stat().st_size
            return data
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"failed {url}: {last}")

def main():
    total_files = 0
    total_bytes = 0
    failures = []
    for cat in CATEGORIES:
        api = f"{BASE}/api/datasets/{REPO}/tree/main/{cat}?recursive=true"
        entries = json.loads(get(api).decode("utf-8"))
        files = [e for e in entries if e.get("type") == "file"]
        est = sum(e.get("size") or 0 for e in files)
        print(f"[{cat}] {len(files)} files, listed size {est/1e6:.1f} MB", flush=True)
        def work(e):
            rel = e["path"]  # e.g. bottle/train/good/000.png
            url = f"{BASE}/datasets/{REPO}/resolve/main/{rel}"
            return rel, get(url, OUT / rel)
        done = 0
        with cf.ThreadPoolExecutor(max_workers=12) as ex:
            futs = {ex.submit(work, e): e for e in files}
            for fut in cf.as_completed(futs):
                e = futs[fut]
                try:
                    rel, size = fut.result()
                    total_files += 1
                    total_bytes += size
                except Exception as exc:  # noqa: BLE001
                    failures.append((e["path"], str(exc)))
                done += 1
                if done % 50 == 0:
                    print(f"[{cat}] {done}/{len(files)}", flush=True)
        print(f"[{cat}] finished {done}/{len(files)}", flush=True)
    print(f"TOTAL files={total_files} bytes={total_bytes} ({total_bytes/1e6:.1f} MB)")
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for p, err in failures:
            print("  ", p, err)
        sys.exit(2)

if __name__ == "__main__":
    main()
