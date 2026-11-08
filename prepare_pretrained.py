#!/usr/bin/env python3
"""准备官方原版 KL-f8 与 ImageNet AlexNet；标准库即可运行。

网络操作只发生在本脚本，训练阶段不隐式下载。
下载支持断点继续，SHA256/大小检查通过后才发布目标文件。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "configs/pretrained_sources.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, record):
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        raise ValueError(f"Missing file or wrong size: {path}")
    actual = sha256(path)
    if actual != record["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}; existing file was not overwritten")
    return actual


def download(record):
    target = ROOT / record["path"]
    if target.exists():
        verify(target, record)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset < record["bytes"]:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(record["url"], headers=headers)
        print(f"Downloading {target.name} ({offset}/{record['bytes']} bytes already present)", flush=True)
        with urllib.request.urlopen(request, timeout=60) as response:
            append = bool(offset and response.status == 206)
            if response.status == 206:
                expected = f"bytes {offset}-"
                if not response.headers.get("Content-Range", "").startswith(expected):
                    raise ValueError("Unexpected HTTP byte range; partial file retained")
            with partial.open("ab" if append else "wb") as output:
                shutil.copyfileobj(response, output)
    verify(partial, record)
    partial.replace(target)
    return target


def prepare(catalog, check_only=False):
    results = {}
    for name, item in catalog["weights"].items():
        target = ROOT / item["file"]["path"]
        if target.exists() or check_only:
            digest = verify(target, item["file"])
        elif "zip_member" in item:
            archive = download(item["download"])
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            # 只读取固定成员，不解压归档中的任意路径。
            with zipfile.ZipFile(archive) as bundle, bundle.open(item["zip_member"]) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output)
            digest = verify(partial, item["file"])
            partial.replace(target)
        else:
            download(item["download"])
            digest = verify(target, item["file"])
        results[name] = {"path": str(target), "sha256": digest,
                         "source_url": item["download"]["url"], "model_identity": item["identity"]}
        print(f"Verified {name}: {target}", flush=True)
    if not check_only:
        (ROOT / "weights/provenance.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="只校验已准备的权重，不联网")
    args = parser.parse_args(argv)
    try:
        prepare(json.loads(CATALOG.read_text()), args.check_only)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(2, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
