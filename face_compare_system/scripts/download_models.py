"""Download only official OpenCV Zoo weights and verify published LFS SHA256."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from face_compare.deep_engine import MODELS


def main():
    directory = ROOT / "models"
    directory.mkdir(exist_ok=True)
    for filename, (folder, digest, size) in MODELS.items():
        target = directory / filename
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            print(f"已校验：{filename}", flush=True)
            continue
        url = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/{folder}/{filename}"
        temporary = target.with_suffix(".download")
        print(f"下载：{filename}", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if temporary.stat().st_size != size or hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"官方校验和不匹配：{filename}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"完成并通过SHA256：{filename}", flush=True)
    for folder in {entry[0] for entry in MODELS.values()}:
        url = f"https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/{folder}/LICENSE"
        with urllib.request.urlopen(url, timeout=30) as response:
            (directory / f"LICENSE-{folder}").write_bytes(response.read())


if __name__ == "__main__":
    main()
