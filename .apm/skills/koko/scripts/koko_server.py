# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 asuka1975
"""koko の採点サーバ。同梱の判定モデルを 1 回だけ読み込み、localhost で採点を受け付ける。

入口は serve_cu130.py / serve_cpu.py（uv のスクリプト。依存の版を固定してある）。koko.py の `server start` が
`uv run --locked --script serve_<種類>.py` で起動する。直接は起動しない想定。

  POST /score  {"texts": ["<1 文>", ...]}  → {"p_yes": [...], "model_id": ...}
  GET  /health → {"ok": true, "protocol": 2, "pid": ..., "model_id": ..., "device": ..., "koko_version": ...}

文への分割はクライアント（koko.py、kokotext.py）が行う。サーバは 1 文ごとの P(Yes) だけを返す。
同じ文には、どの呼び出しでも同じ値を返す（1 文ずつ採点し、結果を覚えておく）。

  --probe  モデルを読まずに、torch などの版と CUDA が使えるかを JSON で出して終わる

環境変数（koko.py が渡す）:
  KOKO_MODEL_DIR     モデルのディレクトリ（manifest.json があるところ）
  KOKO_STATE_FILE    起動したら port と pid を書くファイル
  KOKO_DEVICE        cuda / cpu（既定: CUDA が使えれば cuda）
  KOKO_IDLE_SECONDS  この秒数だけ要求がなければ終了して GPU を空ける（既定: 1800）
  KOKO_VERSION       koko のバージョン（health に載せるだけ）
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROTOCOL = 2


def probe() -> dict:
    import safetensors
    import tokenizers
    import torch
    import transformers

    cuda = torch.cuda.is_available()
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "tokenizers": tokenizers.__version__,
        "cuda": cuda,
        "device_name": torch.cuda.get_device_name(0) if cuda else None,
    }


def serve() -> int:
    import torch

    import kokomodel

    model_dir = Path(os.environ["KOKO_MODEL_DIR"])
    state_file = Path(os.environ["KOKO_STATE_FILE"])
    device = os.environ.get("KOKO_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    idle = float(os.environ.get("KOKO_IDLE_SECONDS", "1800"))
    version = os.environ.get("KOKO_VERSION", "")

    scorer = kokomodel.Scorer(model_dir, device)
    model_id = scorer.manifest["model_id"]
    lock, busy_lock = threading.Lock(), threading.Lock()
    busy, last = [0], [time.monotonic()]
    cache: dict[str, float] = {}

    def p_yes(text: str) -> float:
        if text in cache:
            return cache[text]
        with lock:  # 1 文ごとにロックを取り、並列の要求が交互に進むようにする
            p = scorer.p_yes(text)
        if len(cache) >= 200_000:
            cache.clear()
        cache[text] = p
        return p

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            last[0] = time.monotonic()
            if self.path == "/health":
                self._send(200, {"ok": True, "protocol": PROTOCOL, "pid": os.getpid(), "model_id": model_id, "model_dir": str(model_dir), "device": device, "koko_version": version})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            last[0] = time.monotonic()
            if self.path != "/score":
                self._send(404, {"error": "not found"})
                return
            with busy_lock:
                busy[0] += 1
            try:
                n = int(self.headers.get("Content-Length", "0"))
                texts = json.loads(self.rfile.read(n).decode("utf-8"))["texts"]
                if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                    raise ValueError("texts must be a list of strings")
                self._send(200, {"p_yes": [p_yes(t) for t in texts], "model_id": model_id})
            except Exception as e:  # 要求の誤りはクライアントに返す
                self._send(400, {"error": f"{type(e).__name__}: {e}"})
            finally:
                with busy_lock:
                    busy[0] -= 1
                last[0] = time.monotonic()

        def log_message(self, fmt, *args):
            pass

    state_file.parent.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    port = httpd.server_address[1]
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps({"pid": os.getpid(), "port": port, "protocol": PROTOCOL, "model_id": model_id, "started": time.time()}))
    tmp.replace(state_file)
    print(json.dumps({"event": "ready", "port": port, "model_id": model_id, "device": device}), flush=True)

    def idle_watch():
        while True:
            time.sleep(10)
            with busy_lock:
                n = busy[0]
            if time.monotonic() - last[0] > idle and n == 0:
                print(json.dumps({"event": "idle_exit"}), flush=True)
                httpd.shutdown()
                return

    threading.Thread(target=idle_watch, daemon=True).start()
    try:
        httpd.serve_forever()
    finally:
        try:
            if json.loads(state_file.read_text()).get("pid") == os.getpid():
                state_file.unlink()
        except (OSError, ValueError):
            pass
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--probe"]:
        print(json.dumps(probe()))
        return 0
    if argv:
        raise SystemExit(f"不明な引数: {' '.join(argv)}")
    return serve()
