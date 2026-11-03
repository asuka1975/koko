#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 asuka1975
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.10.0",
#   "transformers==5.17.0",
#   "safetensors==0.8.0",
#   "tokenizers==0.23.3",
#   "numpy==2.5.4",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]
# ///
"""koko の採点サーバを、torch の cu130 版で動かす入口。koko.py が `uv run --locked --script` で起動する。

依存の版は、判定モデルを作った環境と同じにしてある（変えると採点の値が変わることがある）。
すべての版は隣の serve_cu130.py.lock に固定してある。macOS では serve_cpu.py を使う。
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import koko_server  # noqa: E402

raise SystemExit(koko_server.main(sys.argv[1:]))
