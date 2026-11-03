#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 asuka1975
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.10.0", "safetensors==0.8.0", "numpy==2.5.4"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]
# ///
"""量子化した判定モデルのディレクトリを、koko が同梱する形（koko-model-v1）に書き出す（開発用）。

  - model.safetensors を、1 ファイルあたり --max-bytes 以下の shard に分ける（Git LFS の 1 ファイル 2 GB の上限より下にする）
  - quant_config.json から手元の絶対パスを取り除き（手元のパスを配布物に残さない）、形式の名前を koko-quant-v1 にする
  - 全ファイルの大きさと SHA-256、モデルの識別子（model_id）を manifest.json に書く

uv で動かす（torch と safetensors を使う。CPU 版の torch で足りる）:
  uv run tools/shard_model.py <量子化モデルのディレクトリ> <出力先>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import unicodedata
from pathlib import Path

COPY = ("config.json", "decision-inference.json", "tokenizer.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def hidden(ch: str) -> bool:
    o = ord(ch)
    if o < 0x80:
        return False
    cat = unicodedata.category(ch)
    return cat in ("Cc", "Cf", "Co", "Cn", "Cs", "Zs", "Zl", "Zp") or 0xFE00 <= o <= 0xFE0F or 0xE0100 <= o <= 0xE01EF or o == 0x00AD


def escape_hidden(text: str) -> str:
    out = []
    for ch in text:
        if hidden(ch):
            o = ord(ch)
            if o > 0xFFFF:  # サロゲートペアにする
                o -= 0x10000
                out.append(f"\\u{0xD800 + (o >> 10):04x}\\u{0xDC00 + (o & 0x3FF):04x}")
            else:
                out.append(f"\\u{o:04x}")
        else:
            out.append(ch)
    return "".join(out)


def layer_of(name: str) -> int:
    m = re.match(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def local_path(v) -> bool:
    return isinstance(v, str) and v.startswith("/")


def drop_local_paths(obj):
    """手元の絶対パスの値を取り除く（SHA-256 などのほかの値は残す）。"""
    if isinstance(obj, dict):
        return {k: drop_local_paths(v) for k, v in obj.items() if not local_path(v)}
    if isinstance(obj, list):
        return [drop_local_paths(v) for v in obj if not local_path(v)]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--max-bytes", type=int, default=1_000_000_000)
    ap.add_argument("--name", default="saihai-v1-8b-koko-2bit")
    a = ap.parse_args()

    import torch
    from safetensors.torch import load_file, save_file

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        if name.endswith(".json"):
            # 見えない文字（書式制御、ゼロ幅、双方向の制御、異体字セレクタなど）を \\uXXXX に書き換える。
            # JSON としては同じ内容のまま（読み込んだ結果は変わらない）で、APM の安全検査（hidden characters）を通す
            raw = (src / name).read_text(encoding="utf-8")
            esc = escape_hidden(raw)
            if json.loads(esc) != json.loads(raw):
                raise SystemExit(f"{name}: 書き換えで内容が変わりました")
            (out / name).write_text(esc, encoding="utf-8")
        else:
            shutil.copyfile(src / name, out / name)
    qc = drop_local_paths(json.loads((src / "quant_config.json").read_text()))
    qc["format"] = "koko-quant-v1"  # kokomodel.py が読む形（キーの位置は元のまま）
    (out / "quant_config.json").write_text(json.dumps(qc, ensure_ascii=False, indent=1) + "\n")

    tensors = load_file(str(src / "model.safetensors"))
    # 埋め込み・正規化・head を先頭に、その後は層の順に並べ、上限を超える前に次の shard にする
    order = sorted(tensors, key=lambda k: (layer_of(k), k))
    groups: list[list[str]] = [[]]
    size = 0
    for k in order:
        n = tensors[k].numel() * tensors[k].element_size()
        if groups[-1] and size + n > a.max_bytes:
            groups.append([])
            size = 0
        groups[-1].append(k)
        size += n
    shards = [f"model-{i:05d}-of-{len(groups):05d}.safetensors" for i in range(1, len(groups) + 1)]
    for name, keys in zip(shards, groups):
        save_file({k: tensors[k].contiguous() for k in keys}, str(out / name), metadata={"format": "pt"})
    # 書き出した shard を読み戻し、元の重みとビット単位で同じかを確かめる
    merged = {}
    for name in shards:
        merged.update(load_file(str(out / name)))
    if set(merged) != set(tensors) or any(not torch.equal(merged[k], tensors[k]) for k in tensors):
        raise SystemExit("shard を読み戻した重みが元と一致しません")

    files = []
    for p in sorted(out.iterdir()):
        if p.name == "manifest.json" or not p.is_file() or p.suffix not in (".json", ".safetensors"):
            continue
        files.append({"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)})
    model_id = hashlib.sha256("\n".join(f"{f['name']}:{f['sha256']}" for f in files).encode()).hexdigest()
    manifest = {
        "format": "koko-model-v1",
        "name": a.name,
        "model_id": model_id,
        "strip_space": bool(qc.get("strip_space")),
        "shards": shards,
        "files": files,
        "source_sha256": {"model.safetensors": sha256(src / "model.safetensors")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"shards": [(s, (out / s).stat().st_size) for s in shards], "model_id": model_id}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
