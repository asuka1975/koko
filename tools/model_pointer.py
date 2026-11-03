#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 asuka1975
"""判定モデルの取得先（.apm/skills/koko/model.json）を書く（開発用。標準ライブラリだけ）。

koko は判定モデルを同梱せず、初めて使うときに Hugging Face Hub から ~/.cache/koko に取得する。
model.json には、取得先のリポジトリと版（コミットの SHA）、取得するファイルの大きさと SHA-256 を書く。
koko は取得したファイルをこの値と照らす。

  python3 tools/model_pointer.py <モデルのディレクトリ> --repo <user>/<name> --revision <コミットの SHA>

<モデルのディレクトリ> は tools/build_model.py の出力（manifest.json があるところ）。Hugging Face に上げたものと
同じファイルを指定する（1 バイトでも違うと、koko の取得が SHA-256 の照合で止まる）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / ".apm" / "skills" / "koko" / "model.json"
# 動かすのに要るファイル（manifest.json の files）に加えて取得するもの。ライセンスと由来を、重みと一緒に置くため
EXTRA = ("manifest.json", "NOTICE.md", "LICENSE-Apache-2.0.txt")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--repo", required=True, help="Hugging Face のリポジトリ（<user>/<name>）")
    ap.add_argument("--revision", required=True, help="取得する版。コミットの SHA（40 桁）を指定する")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.revision) != 40:
        raise SystemExit("--revision にはコミットの SHA（40 桁）を指定してください（ブランチ名だと中身が変わりうる）")
    d = Path(a.model_dir)
    m = json.loads((d / "manifest.json").read_text())
    names = [f["name"] for f in m["files"]] + [n for n in EXTRA if (d / n).exists()]
    files = []
    for n in names:
        p = d / n
        files.append({"name": n, "bytes": p.stat().st_size, "sha256": sha256(p)})
    for f, g in zip(files, m["files"]):
        if f["sha256"] != g["sha256"]:
            raise SystemExit(f"{f['name']}: manifest.json の SHA-256 と合いません")
    src = {
        "format": "koko-model-source-v1",
        "name": m["name"],
        "model_id": m["model_id"],
        "repo": a.repo,
        "revision": a.revision,
        "files": files,
    }
    Path(a.out).write_text(json.dumps(src, ensure_ascii=False, indent=1) + "\n")
    print(f"{a.out}: {len(files)} ファイル、計 {sum(f['bytes'] for f in files) / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
