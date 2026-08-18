from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
URL = "https://zenodo.org/records/21128640/files/BUSI.zip?download=1"
TARGET = ROOT / "data" / "raw" / "BUSI_zenodo_21128640_complete.zip"
EXPECTED_BYTES = 161_657_476
EXPECTED_MD5 = "c6d336ffa4b0314b1488c7e4c686d7d8"
CHUNK_BYTES = 1024 * 1024


def main() -> int:
    if TARGET.exists():
        raise FileExistsError(
            f"target already exists and will not be overwritten: {TARGET}"
        )
    request = urllib.request.Request(
        URL,
        headers={"User-Agent": "FractalFract-BUSI-research/1.0"},
    )
    md5 = hashlib.md5(usedforsecurity=False)
    total = 0
    with urllib.request.urlopen(request, timeout=120) as response:
        content_length = int(response.headers.get("Content-Length", 0))
        if content_length and content_length != EXPECTED_BYTES:
            raise RuntimeError(
                f"unexpected Content-Length {content_length}, "
                f"expected {EXPECTED_BYTES}"
            )
        with TARGET.open("xb") as handle:
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                handle.write(chunk)
                md5.update(chunk)
                total += len(chunk)
                print(
                    f"downloaded={total}/{EXPECTED_BYTES} "
                    f"({100.0 * total / EXPECTED_BYTES:.1f}%)",
                    flush=True,
                )
    digest = md5.hexdigest()
    if total != EXPECTED_BYTES:
        raise RuntimeError(f"incomplete download: {total}/{EXPECTED_BYTES} bytes")
    if digest != EXPECTED_MD5:
        raise RuntimeError(f"MD5 mismatch: {digest} != {EXPECTED_MD5}")
    print(f"verified target={TARGET}")
    print(f"bytes={total}")
    print(f"md5={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
