#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 asuka1975
"""量子化した判定モデルのディレクトリを、koko が Hugging Face から取得して使う形（koko-model-v1）に書き出す（開発用。標準ライブラリだけ）。

  - model.safetensors は、そのまま写す（中身は変えない）
  - quant_config.json から手元の絶対パスを取り除き（手元のパスを配布物に残さない）、形式の名前を koko-quant-v1 にする
  - JSON の中の見えない文字を \\uXXXX に書き換える（JSON としての内容は変えない。以前 koko のパッケージに同梱していたときに、
    APM の安全検査を通すために入れた書き換えで、配るファイルを変えないために今も同じにしている）
  - 全ファイルの大きさと SHA-256、モデルの識別子（model_id）を manifest.json に書く

  python3 tools/build_model.py <量子化モデルのディレクトリ> <出力先>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import unicodedata
from pathlib import Path

COPY = ("config.json", "decision-inference.json", "tokenizer.json")
WEIGHTS = "model.safetensors"


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
    ap.add_argument("--name", default="saihai-v1-8b-koko-2bit")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        raw = (src / name).read_text(encoding="utf-8")
        esc = escape_hidden(raw)
        if json.loads(esc) != json.loads(raw):
            raise SystemExit(f"{name}: 書き換えで内容が変わりました")
        (out / name).write_text(esc, encoding="utf-8")
    qc = drop_local_paths(json.loads((src / "quant_config.json").read_text()))
    qc["format"] = "koko-quant-v1"  # kokomodel.py が読む形（キーの位置は元のまま）
    (out / "quant_config.json").write_text(json.dumps(qc, ensure_ascii=False, indent=1) + "\n")

    shutil.copyfile(src / WEIGHTS, out / WEIGHTS)
    if sha256(out / WEIGHTS) != sha256(src / WEIGHTS):
        raise SystemExit(f"{WEIGHTS} を写した結果が元と一致しません")

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
        "shards": [WEIGHTS],
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"model_id": model_id, "files": [(f["name"], f["bytes"]) for f in files]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
