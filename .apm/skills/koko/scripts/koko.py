#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 asuka1975
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""koko: 文章を意味のまとまりごとに推敲し、判定モデルの P(Yes) を上げるための道具。

`uv run <このファイル> <サブコマンド>` で動かす（この CLI は標準ライブラリだけで動く）。
採点サーバは、依存の版を固定した uv のスクリプト（serve_cu130.py / serve_cpu.py とそのロック）で動く。

P(Yes) は、判定モデル（saihai-v1-8b を基にファインチューニングし、2bit に量子化したもの）が、
その 1 文を「Wikipedia 由来（人が書いた側）」と判定する確率。0 に近いほど「Claude 由来」と判定されている。

サブコマンド:
  doctor                     動かす準備ができているかを調べる（1 行目に koko-doctor: ready|needs_setup|unavailable）
  setup                      判定モデルを Hugging Face から ~/.cache/koko に取得し（約 2.6 GB）、採点サーバの環境を uv で用意する
  server start|stop|status   採点サーバ（モデルを GPU に 1 回だけ載せる）の起動・停止・確認
  score  [FILE ...|-]        まとまりを採点する。--variants で「===」の行で区切った複数案、--lines で 1 行 1 案
  blocks FILE                文書をブロックに分けて採点し、まとまりの計画の下書きを出す
  split  --plan PLAN         計画どおりにまとまりを切り出し、koko-copy-editor への指示書を書く
  check  ORIG REVISED        原文と推敲案の機械チェック（数値・リンク・コード・括弧書き・長さ・文体）
  assemble --plan PLAN       推敲後のまとまりを文書に戻し、採点し直して報告と記録（record.json）を書く
  verify --workdir W         推敲後の文書が、assemble で記録した本文のままかを確かめる（1 行目に koko-check: …）

環境変数:
  KOKO_TORCH         採点サーバの torch の種類: cu130 / cpu（既定: nvidia-smi があれば cu130。macOS は cpu の入口で、MPS を使える PyPI の torch が入る）
  KOKO_MODEL         取得せずに使うモデルのディレクトリ（manifest.json があるもの）
  KOKO_CACHE_DIR     判定モデルを取得して置く場所（既定: ~/.cache/koko）
  KOKO_MODEL_SOURCE  判定モデルの取得先を書いたファイル（既定: このスキルの model.json）
  HF_ENDPOINT        Hugging Face の URL（既定: https://huggingface.co。ミラーを使うとき）
  KOKO_DEVICE        cuda / mps / cpu（既定: CUDA が使えれば cuda、次に MPS が使えれば mps、どちらもなければ cpu）
  KOKO_IDLE_SECONDS  採点サーバが自動で止まるまでの待ち時間（既定: 1800）
  KOKO_STATE_DIR     採点サーバの状態とログの置き場所（既定: $XDG_RUNTIME_DIR/koko、なければ一時ディレクトリの koko-<uid>）
  KOKO_FAKE_SCORER   1 にすると、モデルを使わず文から決まる仮の値を返す（テスト用）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

sys.dont_write_bytecode = True  # 展開先のスキルのフォルダに __pycache__ を作らない

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kokotext  # noqa: E402

KOKO_VERSION = "0.1.2"
PROTOCOL = 2
SKILL_DIR = HERE.parent
SOURCE_FILE = Path(os.environ.get("KOKO_MODEL_SOURCE") or SKILL_DIR / "model.json").resolve()
CACHE_DIR = Path(os.environ.get("KOKO_CACHE_DIR") or Path.home() / ".cache" / "koko").expanduser().resolve()
LOCAL_MODEL = os.environ.get("KOKO_MODEL")  # 取得せずに、このディレクトリのモデルを使う
ENTRIES = {"cu130": HERE / "serve_cu130.py", "cpu": HERE / "serve_cpu.py"}


def _state_dir() -> Path:
    if os.environ.get("KOKO_STATE_DIR"):
        return Path(os.environ["KOKO_STATE_DIR"])
    if os.environ.get("XDG_RUNTIME_DIR"):
        return Path(os.environ["XDG_RUNTIME_DIR"]) / "koko"
    uid = os.getuid() if hasattr(os, "getuid") else os.environ.get("USERNAME", "user")
    return Path(tempfile.gettempdir()) / f"koko-{uid}"


STATE_DIR = _state_dir()
TARGET = 0.9
VARIANT_SEP = re.compile(r"^===\s*$", re.M)
DEFAULT_PROTECT = [r"^(図|表)\s*\d+\s*[:：.．]"]  # 図表のキャプション（図表の中身と対になっていて、図表を見ずに書き換えると食い違う）
FAKE = os.environ.get("KOKO_FAKE_SCORER") == "1"


def die(msg: str, code: int = 1):
    print(f"koko: {msg}", file=sys.stderr)
    raise SystemExit(code)


# --- モデルと実行環境 ---------------------------------------------------------------------------

_SOURCE: dict | None = None


def source() -> dict:
    """判定モデルの取得先と、取得するファイルの大きさ・SHA-256（model.json）。KOKO_MODEL のときはその manifest.json。"""
    global _SOURCE
    if _SOURCE is None:
        p = Path(LOCAL_MODEL).resolve() / "manifest.json" if LOCAL_MODEL else SOURCE_FILE
        if not p.exists():
            die(f"判定モデルの情報がありません: {p}", 3)
        _SOURCE = json.loads(p.read_text(encoding="utf-8"))
    return _SOURCE


def manifest() -> dict:
    return source()


def model_dir() -> Path:
    """判定モデルを置く場所。既定は ~/.cache/koko/models/<名前>-<model_id の先頭 12 桁>。"""
    if LOCAL_MODEL:
        return Path(LOCAL_MODEL).resolve()
    s = source()
    return CACHE_DIR / "models" / f"{s['name']}-{s['model_id'][:12]}"


def model_id() -> str:
    return "fake" if FAKE else source()["model_id"]


def model_problems(deep: bool = False) -> list[str]:
    """揃っていないファイルの説明。空なら揃っている。deep では SHA-256 も照らす。"""
    problems = []
    d = model_dir()
    for f in source()["files"]:
        p = d / f["name"]
        if not p.exists():
            problems.append(f"{f['name']}: まだ取得していません")
            continue
        size = p.stat().st_size
        if size != f["bytes"]:
            problems.append(f"{f['name']}: {size} バイト（{f['bytes']} バイトのはず）")
        elif deep and sha256_file(p) != f["sha256"]:
            problems.append(f"{f['name']}: SHA-256 が合いません")
    return problems


def _download(url: str, dest: Path, f: dict, quiet: bool) -> None:
    """url を dest に取得する。途中のファイル（.part）があれば続きから取る。SHA-256 が合ったときだけ dest に置く。"""
    import urllib.error
    import urllib.request

    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > f["bytes"]:
        part.unlink()
        have = 0
    h = hashlib.sha256()
    if have:
        with part.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
    if have < f["bytes"]:
        req = urllib.request.Request(url, headers={"User-Agent": f"koko/{KOKO_VERSION}", **({"Range": f"bytes={have}-"} if have else {})})
        try:
            resp = urllib.request.urlopen(req, timeout=60)
        except (urllib.error.URLError, OSError) as e:
            die(f"判定モデルを取得できませんでした（{url}）: {e}。ネットワークを確かめて、やり直してください（取得したところまでは残してあり、続きから取ります）", 3)
        with resp:
            if have and resp.status != 206:  # 続きからの取得に応じなかったので、初めから取る
                have, h = 0, hashlib.sha256()
            mode = "ab" if have else "wb"
            done, shown = have, -1
            with part.open(mode) as out:
                while True:
                    try:
                        block = resp.read(1 << 20)
                    except (urllib.error.URLError, OSError) as e:
                        die(f"判定モデルの取得が途中で止まりました（{f['name']}）: {e}。やり直すと続きから取ります", 3)
                    if not block:
                        break
                    out.write(block)
                    h.update(block)
                    done += len(block)
                    pct = done * 100 // max(f["bytes"], 1)
                    if not quiet and f["bytes"] > (50 << 20) and pct // 10 != shown:
                        shown = pct // 10
                        print(f"koko: {f['name']} {pct}%（{done / 1e9:.2f} / {f['bytes'] / 1e9:.2f} GB）", file=sys.stderr, flush=True)
    if part.stat().st_size != f["bytes"] or h.hexdigest() != f["sha256"]:
        part.unlink(missing_ok=True)
        die(f"取得した {f['name']} の大きさか SHA-256 が、model.json と合いません。やり直しても直らなければ、koko を入れ直してください", 3)
    os.replace(part, dest)


def ensure_model(deep: bool = False, quiet: bool = False) -> Path:
    """判定モデルが揃っていなければ、Hugging Face から取得する。揃ったディレクトリを返す。"""
    d = model_dir()
    if LOCAL_MODEL:
        problems = model_problems(deep)
        if problems:
            die("KOKO_MODEL のモデルに問題があります:\n" + "\n".join(f"  - {p}" for p in problems), 3)
        return d
    s = source()

    def todo(check_sha: bool) -> list[dict]:
        def ok(f: dict) -> bool:
            p = d / f["name"]
            return p.exists() and p.stat().st_size == f["bytes"] and (not check_sha or sha256_file(p) == f["sha256"])

        return [f for f in s["files"] if not ok(f)]

    if not deep and not todo(False):
        return d
    d.mkdir(parents=True, exist_ok=True)
    with open(d / ".lock", "w") as lock:
        try:
            import fcntl  # POSIX だけ。ない環境では、同時に取得すると取り直しになることがある

            fcntl.flock(lock, fcntl.LOCK_EX)
        except ImportError:
            pass
        todo = todo(deep)  # ロック待ちの間に、ほかの呼び出しが取得したかもしれない
        if not todo:
            return d
        endpoint = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
        if not quiet:
            total = sum(f["bytes"] for f in todo)
            print(f"koko: 判定モデルを取得します（{s['repo']}、計 {total / 1e9:.2f} GB）→ {d}", file=sys.stderr, flush=True)
        for f in todo:
            dest = d / f["name"]
            if dest.exists():
                dest.unlink()
            _download(f"{endpoint}/{s['repo']}/resolve/{s['revision']}/{f['name']}", dest, f, quiet)
    return d


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def torch_variant(explicit: str | None = None) -> str:
    v = explicit or os.environ.get("KOKO_TORCH") or ("cu130" if shutil.which("nvidia-smi") else "cpu")
    if v not in ENTRIES:
        die(f"KOKO_TORCH は {' / '.join(ENTRIES)} のどれかにしてください（{v}）")
    return v


def uv_bin() -> str | None:
    return os.environ.get("UV") or shutil.which("uv")


def entry_cmd(variant: str, *extra: str, offline: bool = False) -> list[str]:
    """採点サーバの入口を、ロックファイルどおりの環境で動かすコマンド。"""
    uv = uv_bin()
    if not uv:
        die("uv がありません。https://docs.astral.sh/uv/ の手順で入れてください。", 3)
    return [uv, "run", "--quiet", "--locked", *(["--offline"] if offline else []), "--script", str(ENTRIES[variant]), *extra]


def run(cmd: list, **kw) -> subprocess.CompletedProcess:
    print("$ " + " ".join(str(c) for c in cmd), file=sys.stderr, flush=True)
    r = subprocess.run([str(c) for c in cmd], **kw)
    if r.returncode != 0:
        die(f"コマンドが失敗しました（exit {r.returncode}）: {' '.join(str(c) for c in cmd)}")
    return r


def probe(variant: str, offline: bool) -> dict | None:
    """採点サーバの環境で torch などを読み込めるか。offline では、足りない物を取りに行かない。"""
    try:
        r = subprocess.run(entry_cmd(variant, "--probe", offline=offline), capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def cmd_setup(a) -> int:
    d = ensure_model(deep=True)
    print(json.dumps({"model": str(d), "model_id": model_id()}, ensure_ascii=False))
    if a.model_only:
        return 0
    variant = torch_variant(a.torch)
    uv = uv_bin()
    if not uv:
        die("uv がありません。https://docs.astral.sh/uv/ の手順で入れてください。", 3)
    run([uv, "sync", "--locked", "--script", ENTRIES[variant]])
    info = probe(variant, offline=True)
    if not info:
        die("環境は作れましたが、torch などを読み込めませんでした。`doctor` で確かめてください。")
    print(json.dumps({"variant": variant, **info}, ensure_ascii=False))
    if info.get("mps"):
        print("MPS（Apple Silicon の GPU）で採点します。", file=sys.stderr)
    elif not info["cuda"]:
        print("注意: CUDA も MPS も使えません。採点は CPU で動きます（CPU での速さは確かめていません）。", file=sys.stderr)
    return 0


def doctor_info(deep: bool = False) -> dict:
    info: dict = {"koko_version": KOKO_VERSION, "python": sys.version.split()[0], "skill_dir": str(SKILL_DIR), "notes": []}
    try:
        m = source()
        problems = model_problems(deep)
        missing = bool(problems) and all(p.endswith("まだ取得していません") for p in problems)
        info["model"] = {"dir": str(model_dir()), "name": m.get("name"), "model_id": m["model_id"], "repo": m.get("repo"), "revision": m.get("revision"),
                         "local": bool(LOCAL_MODEL), "ok": not problems, "fetched": not missing, "problems": problems, "deep": deep}
    except SystemExit:
        info["model"] = {"dir": None, "local": bool(LOCAL_MODEL), "ok": False, "fetched": False, "problems": ["判定モデルの情報（model.json）がありません"]}
    variant = torch_variant()
    uv = uv_bin()
    found = probe(variant, offline=True) if uv else None
    info["runtime"] = {"variant": variant, "entry": str(ENTRIES[variant]), "uv": uv, "ready": bool(found), **({k: found[k] for k in ("torch", "transformers")} if found else {})}
    info["gpu"] = {"nvidia_smi": bool(shutil.which("nvidia-smi")), **({"cuda": found["cuda"], "mps": bool(found.get("mps")), "device_name": found["device_name"]} if found else {})}
    h = _health() if info["model"].get("model_id") else None
    info["server"] = {"running": bool(h), **({"pid": h["pid"], "port": h["port"], "device": h.get("device")} if h else {})}
    if not info["model"]["ok"] and (info["model"]["local"] or info["model"]["dir"] is None):
        info["status"] = "unavailable"
        info["notes"].append("判定モデルの情報がないか、KOKO_MODEL のモデルに問題があります。")
    elif not uv:
        info["status"] = "unavailable"
        info["notes"].append("uv がありません。https://docs.astral.sh/uv/ の手順で入れてください。")
    elif not info["model"]["ok"] or not info["runtime"]["ready"]:
        info["status"] = "needs_setup"
        if not info["model"]["ok"]:
            info["notes"].append("`uv run <koko.py> setup` で、判定モデルを Hugging Face から取得してください（約 2.6 GB。取得の途中で止まったときは続きから取ります）。")
        if not info["runtime"]["ready"]:
            info["notes"].append("`uv run <koko.py> setup` で採点サーバの環境を用意してください（torch などを数 GB ダウンロードすることがあります）。")
    else:
        info["status"] = "ready"
    if info["runtime"]["ready"] and info["gpu"].get("cuda") is False:
        if info["gpu"].get("mps"):
            info["notes"].append("CUDA はないので、採点は MPS（Apple Silicon の GPU）で動きます。")
        else:
            info["notes"].append("CUDA も MPS も使えないので、採点は CPU で動きます（CPU での速さは確かめていません）。")
    return info


def cmd_doctor(a) -> int:
    info = doctor_info(a.deep)
    if a.json:
        print(json.dumps(info, ensure_ascii=False, indent=1))
        return 0 if info["status"] == "ready" else 1
    print(f"koko-doctor: {info['status']}")
    m, r, g, s = info["model"], info["runtime"], info["gpu"], info["server"]
    print(f"- koko {info['koko_version']}（{info['skill_dir']}）、python {info['python']}")
    state = "OK" if m["ok"] else ("未取得" if not m.get("fetched") else "問題あり")
    print(f"- モデル: {state} {m.get('name') or ''} {m['dir']}" + ("（SHA-256 も確認）" if m.get("deep") else ""))
    for p in m["problems"]:
        print(f"    - {p}")
    print(f"- 採点サーバの環境: {'あり' if r['ready'] else 'なし'}（torch {r.get('torch') or r['variant']}、uv {'あり' if r['uv'] else 'なし'}）{r['entry']}")
    if g.get("cuda"):
        gpu = "CUDA " + (g.get("device_name") or "")
    elif g.get("mps"):
        gpu = "MPS（Apple Silicon）"
    else:
        gpu = "CUDA・MPS なし" if "cuda" in g else ("nvidia-smi あり" if g["nvidia_smi"] else "nvidia-smi なし")
    print(f"- GPU: {gpu}")
    print(f"- 採点サーバ: {'動作中（pid ' + str(s['pid']) + '）' if s['running'] else '停止中（採点のときに自動で起動）'}")
    for n in info["notes"]:
        print(f"- {n}")
    return 0 if info["status"] == "ready" else 1


# --- 採点サーバ ---------------------------------------------------------------------------------


def _state_paths() -> tuple[Path, Path, Path]:
    stem = f"server-{model_id()[:16]}-p{PROTOCOL}"
    return STATE_DIR / f"{stem}.json", STATE_DIR / f"{stem}.lock", STATE_DIR / f"{stem}.log"


def _http(port: int, path: str, payload: dict | None = None, timeout: float = 900) -> dict:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        die(f"採点サーバがエラーを返しました: {e.read().decode('utf-8', 'replace')}")


def _health() -> dict | None:
    if FAKE:
        return None
    state, _, _ = _state_paths()
    try:
        st = json.loads(state.read_text())
        h = _http(st["port"], "/health", timeout=5)
    except (OSError, ValueError, KeyError, urllib.error.URLError):
        return None
    if not h.get("ok") or h.get("model_id") != model_id() or h.get("protocol") != PROTOCOL:
        return None
    return {**st, **h}


def server_start(quiet: bool = False) -> dict:
    """動いていればその情報を、なければ起動して返す。並列の呼び出しはロックで直列にする。"""
    h = _health()
    if h:
        return h
    state, lockfile, log = _state_paths()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(lockfile, "w") as lock:
        try:
            import fcntl  # POSIX だけ。ない環境では二重起動を防げない

            fcntl.flock(lock, fcntl.LOCK_EX)
        except ImportError:
            pass
        h = _health()  # ロック待ちの間に、ほかの呼び出しが起動したかもしれない
        if h:
            return h
        cmd = entry_cmd(torch_variant())
        mdir = ensure_model(quiet=quiet)
        try:
            state.unlink()
        except OSError:
            pass
        if not quiet:
            print("koko: 採点サーバを起動します（数秒〜十数秒。環境がまだなければ、uv が先に用意します）…", file=sys.stderr, flush=True)
        env = {**os.environ, "KOKO_MODEL_DIR": str(mdir), "KOKO_STATE_FILE": str(state), "KOKO_VERSION": KOKO_VERSION}
        with open(log, "a") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, cwd=str(STATE_DIR), start_new_session=True,
            )
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                die(f"採点サーバが起動に失敗しました（exit {proc.returncode}）。ログ {log} の末尾:\n{log.read_text(errors='replace')[-3000:]}")
            h = _health()
            if h:
                return h
            time.sleep(0.5)
        die(f"採点サーバが 1800 秒で準備できませんでした。ログ: {log}")


def server_stop() -> bool:
    state, _, _ = _state_paths()
    try:
        st = json.loads(state.read_text())
    except (OSError, ValueError):
        return False
    try:
        os.kill(st["pid"], signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(40):
        try:
            os.kill(st["pid"], 0)
        except ProcessLookupError:
            break
        time.sleep(0.25)
    try:
        state.unlink()
    except OSError:
        pass
    return True


def score_sentences(texts: list[str]) -> list[float]:
    """1 文ごとの P(Yes)。同じ文は 1 回だけ送る。"""
    uniq = list(dict.fromkeys(texts))
    if not uniq:
        return []
    if FAKE:
        table = {t: int(hashlib.sha256(t.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF for t in uniq}
    else:
        h = server_start()
        try:
            probs = _http(h["port"], "/score", {"texts": uniq})["p_yes"]
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            # サーバが待ち時間切れで止まった、または再起動した直後。1 回だけ起動し直してやり直す
            h = server_start()
            probs = _http(h["port"], "/score", {"texts": uniq})["p_yes"]
        table = dict(zip(uniq, probs))
    return [table[t] for t in texts]


def score_chunks(texts: list[str]) -> list[dict]:
    split = [kokotext.extract(t) for t in texts]
    probs = score_sentences([s["text"] for ss in split for s in ss])
    out, k = [], 0
    for ss in split:
        rows = []
        for s in ss:
            rows.append({**s, "p_yes": probs[k]})
            k += 1
        out.append({"n": len(rows), "mean": sum(r["p_yes"] for r in rows) / len(rows) if rows else None, "sentences": rows})
    return out


# --- 表示 ---------------------------------------------------------------------------------------


def fmt_mean(m: float | None) -> str:
    return "  —  " if m is None else f"{m:.3f}"


def print_result(label: str, res: dict, target: float) -> None:
    m = res["mean"]
    gap = "" if m is None else ("  ✓ 目標達成" if m >= target else f"  目標 {target} まで あと {target - m:.3f}")
    print(f"== {label}  平均 P(Yes) {fmt_mean(m)}（{res['n']} 文）{gap}")
    for i, s in enumerate(res["sentences"], 1):
        print(f"  {i:>2}. {s['p_yes']:.3f}  {s['shown']}")
        if s["text"] != s["shown"]:
            print(f"        └ 採点された形（短い括弧書きを除去）: {s['text']}")
    print()


def cmd_score(a) -> int:
    items: list[tuple[str, str]] = []
    files = a.files or ["-"]
    for p in files:
        name = "stdin" if p == "-" else p
        text = sys.stdin.read() if p == "-" else Path(p).read_text(encoding="utf-8")
        prefix = f"{name} " if len(files) > 1 else ""
        if a.variants:
            parts = [x.strip("\n") for x in VARIANT_SEP.split(text) if x.strip()]
            items += [(f"{prefix}案{i}", x) for i, x in enumerate(parts, 1)]
        elif a.lines:
            items += [(f"{prefix}案{i}", x.strip()) for i, x in enumerate((y for y in text.splitlines() if y.strip()), 1)]
        else:
            items.append((name, text))
    if not items:
        die("採点する文章がありません")
    results = score_chunks([t for _, t in items])
    if a.json:
        print(json.dumps({"model_id": model_id(), "target": a.target, "results": [{"label": l, **r} for (l, _), r in zip(items, results)]}, ensure_ascii=False, indent=1))
        return 0
    for (label, _), r in zip(items, results):
        print_result(label, r, a.target)
    if len(items) > 1:
        ranked = sorted(zip(items, results), key=lambda x: -(x[1]["mean"] if x[1]["mean"] is not None else -1))
        print("== 平均の高い順: " + "、".join(f"{l} {fmt_mean(r['mean'])}" for (l, _), r in ranked))
    return 0


def cmd_server(a) -> int:
    if a.action == "start":
        h = server_start()
        print(json.dumps({"running": True, "pid": h["pid"], "port": h["port"], "device": h.get("device"), "model_id": h["model_id"]}, ensure_ascii=False))
    elif a.action == "stop":
        print(json.dumps({"stopped": server_stop()}))
    else:
        h = _health()
        print(json.dumps({"running": bool(h), **({"pid": h["pid"], "port": h["port"], "device": h.get("device"), "model_id": h["model_id"]} if h else {})}, ensure_ascii=False))
    return 0


# --- 文書のブロック -----------------------------------------------------------------------------

FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
HR = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
LIST_LINE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
CAPTION = re.compile(r"^(図|表)\s*\d+\s*[:：.．]")
PROSE = ("paragraph", "list", "quote", "caption")


def parse_blocks(text: str, protect: list[str] | None = None) -> list[dict]:
    """空行で区切ったブロック。行番号は 1 始まり。

    見出し・表・画像・コード・区切り線・front matter・HTML は固定（まとまりに入れない）。
    protect の正規表現のどれかに 1 行目が合う地の文も protected として固定する。
    """
    pats = [re.compile(p) for p in (protect or [])]
    lines = text.split("\n")
    n = len(lines)
    out: list[dict] = []
    section = ""
    i = 0

    def add(kind, s, e):
        nonlocal section
        body = "\n".join(lines[s : e + 1])
        if kind == "heading":
            section = body.strip().lstrip("#").strip()
        if kind in PROSE and any(p.search(lines[s].strip()) for p in pats):
            kind = "protected"
        out.append({"id": f"b{len(out) + 1:02d}", "kind": kind, "start": s + 1, "end": e + 1, "text": body, "section": section})

    if n and lines[0].strip() == "---":
        j = next((k for k in range(1, n) if lines[k].strip() in ("---", "...")), None)
        if j is not None:
            add("frontmatter", 0, j)
            i = j + 1
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        s = i
        m = FENCE.match(line)
        if m:
            j = next((k for k in range(i + 1, n) if lines[k].lstrip().startswith(m.group(1))), n - 1)
            add("code", s, j)
            i = j + 1
            continue
        if HEADING.match(line):
            add("heading", s, s)
            i += 1
            continue
        if line.lstrip().startswith("<!--"):
            j = next((k for k in range(i, n) if "-->" in lines[k]), n - 1)
            add("html", s, j)
            i = j + 1
            continue
        j = i
        while j + 1 < n and lines[j + 1].strip() and not FENCE.match(lines[j + 1]) and not HEADING.match(lines[j + 1]):
            j += 1
        blk = lines[s : j + 1]
        first = blk[0].lstrip()
        if all(x.lstrip().startswith("|") for x in blk):
            kind = "table"
        elif len(blk) == 1 and HR.match(blk[0]):
            kind = "hr"
        elif first.startswith("![") and all(x.lstrip().startswith("![") or not x.strip() for x in blk):
            kind = "image"
        elif first.startswith("<") and not first.startswith("<http"):
            kind = "html"
        elif CAPTION.match(first):
            kind = "caption"
        elif all(x.lstrip().startswith(">") for x in blk):
            kind = "quote"
        elif LIST_LINE.match(blk[0]) and all(LIST_LINE.match(x) or x.startswith((" ", "\t")) for x in blk):
            kind = "list"
        else:
            kind = "paragraph"
        add(kind, s, j)
        i = j + 1
    return out


def suggest_chunks(blocks: list[dict], counts: dict[str, int]) -> list[dict]:
    """機械的な下書き: 導入の段落と直後の箇条書きをまとめ、1 文だけの段落は同じ節の隣（後ろ優先）とまとめる。"""
    groups: list[list[dict]] = []
    prev = None
    for b in blocks:
        if b["kind"] not in PROSE:
            prev = None
            continue
        if prev is not None and b["kind"] in ("list", "quote") and prev["kind"] == "paragraph" and groups and groups[-1][-1] is prev and b["section"] == prev["section"]:
            groups[-1].append(b)
        else:
            groups.append([b])
        prev = b

    def adjacent(g1, g2):
        a, b = g1[-1], g2[0]
        return not [x for x in blocks if a["end"] < x["start"] < b["start"]] and a["section"] == b["section"]

    changed = True
    while changed:
        changed = False
        for j, g in enumerate(groups):
            if g[0]["kind"] == "caption" or sum(counts.get(b["id"], 0) for b in g) > 1:
                continue
            for k in (j + 1, j - 1):
                if 0 <= k < len(groups) and groups[k][0]["kind"] != "caption":
                    lo, hi = sorted((j, k))
                    if adjacent(groups[lo], groups[hi]):
                        groups[lo] = groups[lo] + groups[hi]
                        del groups[hi]
                        changed = True
                        break
            if changed:
                break
    return [
        {"id": f"c{i:02d}", "lines": [g[0]["start"], g[-1]["end"]], "section": g[0]["section"], "intent": "", "prev_constraints": "", "next_constraints": ""}
        for i, g in enumerate(groups, 1)
    ]


def cmd_blocks(a) -> int:
    src = Path(a.file).resolve()
    text = src.read_text(encoding="utf-8")
    protect = ([] if a.no_default_protect else list(DEFAULT_PROTECT)) + (a.protect or [])
    blocks = parse_blocks(text, protect)
    prose = [b for b in blocks if b["kind"] in PROSE]
    res = score_chunks([b["text"] for b in prose]) if prose else []
    by_id = {b["id"]: r for b, r in zip(prose, res)}
    counts = {k: r["n"] for k, r in by_id.items()}
    total = [s["p_yes"] for r in res for s in r["sentences"]]
    plan_chunks = suggest_chunks(blocks, counts)
    for c in plan_chunks:
        ps = [s["p_yes"] for b in prose if c["lines"][0] <= b["start"] <= c["lines"][1] for s in by_id[b["id"]]["sentences"]]
        c["_n"], c["_mean"] = len(ps), (sum(ps) / len(ps) if ps else None)
    if a.plan_out:
        plan = {"source": str(src), "target": a.target, "style": a.style or "", "protect": protect, "chunks": [{k: v for k, v in c.items() if not k.startswith("_")} for c in plan_chunks]}
        Path(a.plan_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.plan_out).write_text(json.dumps(plan, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if a.json:
        print(json.dumps({
            "source": str(src), "model_id": model_id(), "sentences": len(total), "mean": sum(total) / len(total) if total else None,
            "blocks": [{k: b[k] for k in ("id", "kind", "start", "end", "section")} | ({"n": by_id[b["id"]]["n"], "mean": by_id[b["id"]]["mean"], "sentences": by_id[b["id"]]["sentences"]} if b["id"] in by_id else {}) for b in blocks],
            "suggested_chunks": [{"id": c["id"], "lines": c["lines"], "section": c["section"], "n": c["_n"], "mean": c["_mean"]} for c in plan_chunks],
            "plan": str(Path(a.plan_out).resolve()) if a.plan_out else None,
        }, ensure_ascii=False, indent=1))
        return 0
    print(f"文書: {src}（{len(text.splitlines())} 行、文 {len(total)}、平均 P(Yes) {fmt_mean(sum(total) / len(total) if total else None)}）\n")
    for b in blocks:
        head = b["text"].split("\n")[0].strip()[:56]
        if b["id"] in by_id:
            r = by_id[b["id"]]
            print(f"{b['id']} {b['kind']:9s} {b['start']:>4}-{b['end']:<4} {r['n']:>2}文 {fmt_mean(r['mean'])}  {head}")
            if a.sentences:
                for s in r["sentences"]:
                    print(f"      {s['p_yes']:.3f}  {s['shown']}")
        else:
            print(f"{b['id']} {b['kind']:9s} {b['start']:>4}-{b['end']:<4}  （固定）  {head}")
    print("\nまとまりの下書き（機械的な案。意味の切れ目に合わせて直すこと）:")
    for c in plan_chunks:
        print(f"  {c['id']} 行 {c['lines'][0]}-{c['lines'][1]}  {c['_n']:>2}文 平均 {fmt_mean(c['_mean'])}  [{c['section'][:20]}]")
    if a.plan_out:
        print(f"\n計画の下書きを書きました: {Path(a.plan_out).resolve()}")
    return 0


# --- 計画と切り出し -----------------------------------------------------------------------------


def plan_protect(plan: dict) -> list[str]:
    return list(plan["protect"]) if "protect" in plan else list(DEFAULT_PROTECT)


def validate_plan(plan: dict, blocks: list[dict]) -> list[str]:
    errs = []
    chunks = plan.get("chunks") or []
    if not chunks:
        return ["chunks が空です"]
    starts = {b["start"] for b in blocks}
    ends = {b["end"] for b in blocks}
    seen, last_end = set(), 0
    for c in chunks:
        cid = c.get("id", "?")
        if cid in seen:
            errs.append(f"{cid}: id が重複しています")
        seen.add(cid)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(cid)):
            errs.append(f"{cid}: id は英数字・-・_ だけにしてください")
        try:
            s, e = (int(x) for x in c["lines"])
        except (KeyError, TypeError, ValueError):
            errs.append(f"{cid}: lines は [開始行, 終了行] にしてください")
            continue
        if s not in starts or e not in ends or s > e:
            errs.append(f"{cid}: 行 {s}-{e} がブロックの境目（`blocks` の行番号）と合っていません")
            continue
        fixed = [b for b in blocks if s <= b["start"] and b["end"] <= e and b["kind"] not in PROSE]
        if fixed:
            errs.append(f"{cid}: 固定のブロック（見出し・表・コード・保護した行など）を含んでいます（{', '.join(b['id'] + ' ' + b['kind'] for b in fixed)}）")
        if s <= last_end:
            errs.append(f"{cid}: 前のまとまりと重なっているか、順序が逆です")
        last_end = max(last_end, e)
        if not str(c.get("intent", "")).strip():
            errs.append(f"{cid}: intent（この複数文で伝えたいこと）が空です")
    return errs


def chunk_text(lines: list[str], c: dict) -> str:
    s, e = c["lines"]
    return "\n".join(lines[s - 1 : e]).strip("\n")


BRIEF = """# まとまり {id} の校閲指示

koko-copy-editor への指示書です。下の「原文」を推敲し、まとまり内の文の平均 P(Yes) を {target} 以上にしてください。

## 入出力
- 原文のファイル: {chunk_path}（元の文書の {lines}。節「{section}」）
- 推敲後の本文の書き出し先: {revised_path}（まとまりの本文だけ。前置きや説明は書かない）
- 作業メモの書き出し先: {notes_path}
- 作業用のファイルの置き場所: {work_dir}/（`{id}-` で始まる名前にする）
- 採点: `uv run {koko} score <ファイル>`（複数案は `--variants`、1 行 1 案は `--lines`）
- 機械チェック: `uv run {koko} check {chunk_path} <ファイル>`
- 文書全体（原文）: {source_path}

## 守る条件
- 目標: 平均 P(Yes) ≥ {target}
- 文体: {style}

### この複数文で伝えたいこと
{intent}

### 前のまとまりと内容的に整合させる内容
{prev}

### 次のまとまりと内容的に整合させる内容
{next}
{notes}
## 原文（{n} 文、平均 P(Yes) {mean}）
<chunk>
{text}
</chunk>

文ごとの P(Yes):
{scores}

## 文書の流れ（各まとまりの伝えたいこと。← がこのまとまり）
{outline}

## 文脈（原文。書き換えない）
### 前のまとまり{prev_id}
<previous>
{prev_text}
</previous>

### このまとまりと前後の間にある見出しなど（書き換えない）
{between}

### 次のまとまり{next_id}
<next>
{next_text}
</next>
"""

UNSPECIFIED = "（未指定。前後の文脈を読み、自分で補完すること。補完した内容は作業メモに書く）"


def cmd_split(a) -> int:
    plan_path = Path(a.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    src = Path(plan["source"]).resolve()
    text = src.read_text(encoding="utf-8")
    protect = plan_protect(plan)
    lines, blocks = text.split("\n"), parse_blocks(text, protect)
    errs = validate_plan(plan, blocks)
    if errs:
        print("計画に問題があります:\n" + "\n".join(f"- {x}" for x in errs), file=sys.stderr)
        return 2
    workdir = Path(a.workdir).resolve()
    cdir = workdir / "chunks"
    if cdir.exists() and any(cdir.iterdir()):
        if not a.force:
            print(f"{cdir} に前の切り出しがあります。やり直すなら --force を付けてください（今の chunks/ は退避します）。", file=sys.stderr)
            return 2
        backup = workdir / f"chunks.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        cdir.rename(backup)
        print(f"前の chunks/ を {backup} に退避しました。", file=sys.stderr)
    (cdir / "work").mkdir(parents=True, exist_ok=True)
    (workdir / "source.md").write_text(text, encoding="utf-8")
    target = float(plan.get("target") or TARGET)
    plan = {**plan, "source": str(src), "target": target, "protect": protect, "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    (workdir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    chunks = plan["chunks"]
    texts = [chunk_text(lines, c) for c in chunks]
    res = score_chunks(texts)
    covered = set()
    for c in chunks:
        covered.update(range(c["lines"][0], c["lines"][1] + 1))
    left = [b for b in blocks if b["kind"] in PROSE and b["start"] not in covered]
    for i, (c, t, r) in enumerate(zip(chunks, texts, res)):
        cid = c["id"]
        (cdir / f"{cid}.md").write_text(t + "\n", encoding="utf-8")
        (cdir / f"{cid}.orig.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        prev_c = chunks[i - 1] if i > 0 else None
        next_c = chunks[i + 1] if i + 1 < len(chunks) else None
        lo = prev_c["lines"][1] if prev_c else 0
        hi = next_c["lines"][0] if next_c else len(lines) + 1
        between = []
        for b in blocks:
            before = lo < b["start"] < c["lines"][0]
            after_ = c["lines"][1] < b["start"] < hi
            if before or after_:
                label = b["kind"] if b["kind"] not in PROSE else f"{b['kind']}（推敲しない地の文）"
                between.append(f"- {'前' if before else '後'} {b['start']} 行の {label}: {b['text'].splitlines()[0][:80]}")
        s, e = c["lines"]
        brief = BRIEF.format(
            id=cid,
            target=target,
            chunk_path=cdir / f"{cid}.md",
            revised_path=cdir / f"{cid}.revised.md",
            notes_path=cdir / f"{cid}.notes.md",
            work_dir=cdir / "work",
            koko=Path(__file__).resolve(),
            source_path=workdir / "source.md",
            lines=f"{s} 行" if s == e else f"{s}〜{e} 行",
            section=c.get("section", ""),
            style=c.get("style") or plan.get("style") or "原文の文体（です・ます調／である調）に合わせる",
            intent=str(c["intent"]).strip(),
            prev=str(c.get("prev_constraints") or "").strip() or UNSPECIFIED,
            next=str(c.get("next_constraints") or "").strip() or UNSPECIFIED,
            notes="".join(
                f"\n### {title}\n{str(body).strip()}\n"
                for title, body in (("文書全体にかかる補足", plan.get("notes")), ("このまとまりへの補足", c.get("notes")))
                if str(body or "").strip()
            ),
            n=r["n"],
            mean=fmt_mean(r["mean"]),
            text=t,
            scores="\n".join(f"{k}. {x['p_yes']:.3f} {x['shown']}" for k, x in enumerate(r["sentences"], 1)),
            outline="\n".join(f"- {x['id']}（節「{x.get('section', '')}」）: {' '.join(str(x['intent']).split())}" + ("  ←" if x is c else "") for x in chunks),
            prev_id=f" {prev_c['id']}（{cdir / (prev_c['id'] + '.md')}）" if prev_c else "",
            prev_text=texts[i - 1] if prev_c else "（なし。文書の最初のまとまり）",
            next_id=f" {next_c['id']}（{cdir / (next_c['id'] + '.md')}）" if next_c else "",
            next_text=texts[i + 1] if next_c else "（なし。文書の最後のまとまり）",
            between="\n".join(between) or "（なし）",
        )
        (cdir / f"{cid}.brief.md").write_text(brief, encoding="utf-8")
    rows = [{"id": c["id"], "lines": c["lines"], "n": r["n"], "mean": r["mean"], "met": r["mean"] is not None and r["mean"] >= target, "brief": str(cdir / f"{c['id']}.brief.md")} for c, r in zip(chunks, res)]
    if a.json:
        print(json.dumps({"workdir": str(workdir), "target": target, "chunks": rows, "uncovered": [{"id": b["id"], "start": b["start"], "end": b["end"]} for b in left]}, ensure_ascii=False, indent=1))
        return 0
    print(f"作業ディレクトリ: {workdir}")
    for x in rows:
        print(f"  {x['id']} 行 {x['lines'][0]}-{x['lines'][1]}  {x['n']:>2}文 平均 {fmt_mean(x['mean'])}{'（目標以上）' if x['met'] else ''}  指示書 {x['brief']}")
    if left:
        print("\n注意: どのまとまりにも入っていない地の文があります（推敲されず、そのまま残ります）:")
        for b in left:
            print(f"  {b['id']} 行 {b['start']}-{b['end']}: {b['text'].splitlines()[0][:60]}")
    return 0


# --- 機械チェック -------------------------------------------------------------------------------

NUM = re.compile(r"\d+(?:[.,]\d+)*")
URLS = re.compile(r"https?://[^\s)>\]]+")
CODE = re.compile(r"`[^`\n]+`")
QUOTE = re.compile(r"「([^「」]{1,80})」")
PAREN = re.compile(r"[（(][^（）()]*[）)]")
POLITE = re.compile(r"(です|ます|ました|でした|ません|ましょう|でしょう|ください)[。！？!?」』）)]*$")


def _plain(s: str) -> str:
    return re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", unicodedata.normalize("NFKC", s))


def check(original: str, revised: str) -> tuple[list[str], list[str]]:
    """(errors, warnings)。errors は必ず直す。warnings は、意図して変えたのなら作業メモに理由を書く。"""
    errs, warns = [], []
    o, v = _plain(original), _plain(revised)
    for line in revised.split("\n"):
        if HEADING.match(line) or line.lstrip().startswith(("|", "```", "~~~", "![")):
            errs.append(f"見出し・表・コード・画像は作らないでください: {line.strip()[:40]}")
            break
    if any(tag in revised for tag in ("<revised>", "</revised>", "<chunk>", "</chunk>")):
        errs.append("タグ（<revised> など）が本文に残っています")
    if not v.strip():
        errs.append("本文が空です")
    lost_urls = sorted(set(URLS.findall(original)) - set(URLS.findall(revised)))
    if lost_urls:
        errs.append("リンク先が消えています: " + "、".join(lost_urls))
    lost_code = sorted(set(CODE.findall(original)) - set(CODE.findall(revised)))
    if lost_code:
        warns.append("インラインコードが消えています: " + "、".join(lost_code[:10]))
    lost_num = sorted(set(NUM.findall(PAREN.sub("", o))) - set(NUM.findall(v)), key=lambda x: (len(x), x))
    if lost_num:
        warns.append("括弧の外にあった数値が消えています: " + "、".join(lost_num[:12]))
    lost_q = sorted(set(QUOTE.findall(o)) - set(QUOTE.findall(v)))
    if lost_q:
        warns.append("「」で括った表現が消えています: " + "、".join(f"「{x}」" for x in lost_q[:8]))
    po = sum(len(m) - 2 for m in PAREN.findall(o) if len(m) - 2 <= kokotext.MAX_PARENTHETICAL_CHARS)
    pv = sum(len(m) - 2 for m in PAREN.findall(v) if len(m) - 2 <= kokotext.MAX_PARENTHETICAL_CHARS)
    if pv - po > 15:
        warns.append(f"60 字以下の括弧書きが {po} 字から {pv} 字に増えています（括弧書きは採点前に取り除かれます。本文を括弧に逃がしていないか確認）")
    so, sv = [s["shown"] for s in kokotext.extract(original)], [s["shown"] for s in kokotext.extract(revised)]
    lo, lv = sum(map(len, so)), sum(map(len, sv))
    if lo and not 0.6 <= lv / lo <= 1.6:
        warns.append(f"長さが原文の {lv / lo:.0%} です（60〜160% の外）")
    if so and sv:
        def polite(x: str) -> bool:
            return bool(POLITE.search(PAREN.sub("", x).strip()))

        ro, rv = sum(map(polite, so)) / len(so), sum(map(polite, sv)) / len(sv)
        if abs(ro - rv) > 0.3:
            warns.append(f"です・ます調の文の割合が {ro:.0%} から {rv:.0%} に変わっています")
        if len(so) >= 3 and len(sv) <= max(1, len(so) // 2):
            warns.append(f"文の数が {len(so)} から {len(sv)} に減っています（低い文の内容をほかの文に詰め込んでいないか、伝えたいことが落ちていないか確認）")
    return errs, warns


def cmd_check(a) -> int:
    o = Path(a.original).read_text(encoding="utf-8")
    v = sys.stdin.read() if a.revised == "-" else Path(a.revised).read_text(encoding="utf-8")
    errs, warns = check(o, v)
    if a.json:
        print(json.dumps({"errors": errs, "warnings": warns}, ensure_ascii=False, indent=1))
        return 1 if errs else 0
    if not errs and not warns:
        print("機械チェック: 問題なし")
    for x in errs:
        print(f"エラー: {x}")
    for x in warns:
        print(f"注意: {x}")
    return 1 if errs else 0


# --- 組み立てと記録 -----------------------------------------------------------------------------


def cmd_assemble(a) -> int:
    plan_path = Path(a.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    workdir = plan_path.parent
    cdir = workdir / "chunks"
    src_text = (workdir / "source.md").read_text(encoding="utf-8")
    lines = src_text.split("\n")
    target = float(plan.get("target") or TARGET)
    chunks = plan["chunks"]
    originals = [chunk_text(lines, c) for c in chunks]
    revised = []
    for c in chunks:
        p = cdir / f"{c['id']}.revised.md"
        revised.append(p.read_text(encoding="utf-8").strip("\n") if p.exists() else None)
    res = score_chunks(originals + [r if r is not None else o for r, o in zip(revised, originals)])
    before, after = res[: len(chunks)], res[len(chunks) :]
    # 文書を組み立てる（まとまりの外は 1 文字も変えない）
    out_lines: list[str] = []
    k = 0
    starts = {c["lines"][0]: (i, c) for i, c in enumerate(chunks)}
    while k < len(lines):
        if k + 1 in starts:
            i, c = starts[k + 1]
            out_lines.extend((revised[i] if revised[i] is not None else originals[i]).split("\n"))
            k = c["lines"][1]
        else:
            out_lines.append(lines[k])
            k += 1
    source = Path(plan["source"]).resolve()
    out = Path(a.out).resolve() if a.out else source
    if out == source and out.exists() and not a.force:
        # split のあとに原文が書き換えられていたら、上書きでその変更を消さないように止める
        cur = hashlib.sha256(out.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        known = {hashlib.sha256(src_text.encode("utf-8")).hexdigest()}
        try:
            known.add(json.loads((workdir / "record.json").read_text(encoding="utf-8"))["out_sha256"])
        except (OSError, ValueError, KeyError):
            pass
        if cur not in known:
            die(f"原文が split のあとに書き換えられています（{out}）。上書きすると、その変更が消えます。"
                "split からやり直すか、変更を消してよいなら --force を付けてください", 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(out_lines)
    out.write_text(body, encoding="utf-8")
    # 報告と記録
    rows, detail, rec_chunks = [], [], []
    met = done = 0
    for c, o, rv, b, af in zip(chunks, originals, revised, before, after):
        status = "unrevised"
        errs, warns = ([], [])
        if rv is not None:
            done += 1
            ok = af["mean"] is not None and af["mean"] >= target
            met += ok
            status = "met" if ok else "not_met"
            errs, warns = check(o, rv)
            notes = cdir / f"{c['id']}.notes.md"
            if errs or warns or notes.exists():
                detail.append(f"### {c['id']}")
                detail += [f"- エラー: {x}" for x in errs] + [f"- 注意: {x}" for x in warns]
                if notes.exists():
                    detail.append(f"- 作業メモ: {notes}")
                detail.append("")
        elif b["mean"] is not None and b["mean"] >= target:
            status = "already_met"
        label = {"met": "達成", "not_met": "未達", "unrevised": "未推敲", "already_met": "推敲前から達成"}[status]
        rows.append(f"| {c['id']} | {c['lines'][0]}–{c['lines'][1]} | {c.get('section', '')} | {b['n']} → {af['n']} | {fmt_mean(b['mean'])} → {fmt_mean(af['mean'])} | {label} | {f'{len(errs)} / {len(warns)}' if rv is not None else ''} |")
        rec_chunks.append({"id": c["id"], "lines": c["lines"], "section": c.get("section", ""), "status": status, "n_before": b["n"], "n_after": af["n"], "mean_before": b["mean"], "mean_after": af["mean"], "check_errors": errs, "check_warnings": warns})
    all_b = [s["p_yes"] for r in before for s in r["sentences"]]
    all_a = [s["p_yes"] for r in after for s in r["sentences"]]
    mb = sum(all_b) / len(all_b) if all_b else None
    ma = sum(all_a) / len(all_a) if all_a else None
    record = {
        "format": "koko-record-v1",
        "koko_version": KOKO_VERSION,
        "model_id": model_id(),
        "model_name": "fake" if FAKE else manifest().get("name"),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": plan["source"],
        "source_sha256": hashlib.sha256(src_text.encode("utf-8")).hexdigest(),
        "out": str(out),
        "out_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "target": target,
        "summary": {"chunks": len(chunks), "revised": done, "met": met, "already_met": sum(x["status"] == "already_met" for x in rec_chunks), "mean_before": mb, "mean_after": ma, "sentences_before": len(all_b), "sentences_after": len(all_a)},
        "chunks": rec_chunks,
    }
    (workdir / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (workdir / "scores.json").write_text(json.dumps([{"id": c["id"], "before": b, "after": af} for c, b, af in zip(chunks, before, after)], ensure_ascii=False, indent=1), encoding="utf-8")
    report = [
        "# koko の推敲結果",
        "",
        f"- 元の文書: {plan['source']}",
        f"- 推敲後の文書: {out}",
        f"- 判定モデル: {record['model_name']}（{record['model_id'][:16]}）",
        f"- 目標: まとまり内の文の平均 P(Yes) ≥ {target}",
        f"- 推敲したまとまり: {done} / {len(chunks)}、目標を達成: {met} / {done}（推敲前から達成: {record['summary']['already_met']}）",
        f"- 文書全体の平均 P(Yes)（まとまり内の文）: {fmt_mean(mb)} → {fmt_mean(ma)}",
        "",
        "| まとまり | 行（原文） | 節 | 文の数 | 平均 P(Yes) | 目標 | チェック（エラー / 注意） |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
    ]
    if detail:
        report += ["## チェックと作業メモ", "", *detail]
    (workdir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if a.json:
        print(json.dumps(record, ensure_ascii=False, indent=1))
        return 0
    print("\n".join(report))
    print(f"報告: {workdir / 'report.md'}\n記録: {workdir / 'record.json'}\n文ごとの値: {workdir / 'scores.json'}")
    return 0


VERIFY_CODES = {"ok": 0, "modified": 1, "no_record": 3, "missing_final": 4, "model_changed": 5}


def cmd_verify(a) -> int:
    """推敲後の文書が、assemble で記録した本文のままか。読むだけで、何も書かない。"""
    rec_path = Path(a.workdir).resolve() / "record.json"
    result: dict = {"record": str(rec_path)}
    if not rec_path.exists():
        status = "no_record"
        result["message"] = "koko の記録（record.json）がありません。assemble を実行してください。"
    else:
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        final = Path(a.final).resolve() if a.final else Path(rec["out"])
        result.update({"final": str(final), "summary": rec["summary"], "target": rec["target"]})
        if not final.exists():
            status = "missing_final"
            result["message"] = f"推敲後の文書がありません: {final}"
        elif hashlib.sha256(final.read_text(encoding="utf-8").encode("utf-8")).hexdigest() != rec["out_sha256"]:
            status = "modified"
            result["message"] = "推敲後の文書が、記録した本文と違います（assemble のあとに書き換えられた）。assemble をやり直してください。"
        elif rec["model_id"] != model_id():
            status = "model_changed"
            result["message"] = "記録したときの判定モデルと、いまのモデルが違います。assemble をやり直してください。"
        else:
            status = "ok"
            s = rec["summary"]
            result["message"] = f"記録と合いました。目標達成 {s['met']} / {s['revised']}（推敲前から達成 {s['already_met']}）、平均 P(Yes) {fmt_mean(s['mean_before'])} → {fmt_mean(s['mean_after'])}。"
    result["status"] = status
    if a.json:
        print(json.dumps(result, ensure_ascii=False, indent=1))
    else:
        print(f"koko-check: {status}")
        print(result["message"])
    return VERIFY_CODES[status]


# --- main ---------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"koko {KOKO_VERSION}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("doctor", help="動かす準備ができているかを調べる")
    d.add_argument("--deep", action="store_true", help="モデルの SHA-256 も確かめる（数秒かかる）")
    d.add_argument("--json", action="store_true")
    st = sub.add_parser("setup", help="実行環境を作る")
    st.add_argument("--torch", choices=sorted(ENTRIES), help="torch の種類（既定: nvidia-smi があれば cu130、なければ cpu）")
    st.add_argument("--model-only", action="store_true", help="判定モデルの取得だけをする（採点サーバの環境は作らない）")
    s = sub.add_parser("server", help="採点サーバの起動・停止・確認")
    s.add_argument("action", choices=["start", "stop", "status"])
    sc = sub.add_parser("score", help="まとまりを採点する")
    sc.add_argument("files", nargs="*", help="Markdown のファイル。省略か - で標準入力")
    g = sc.add_mutually_exclusive_group()
    g.add_argument("--variants", action="store_true", help="「===」だけの行で区切った複数の案を別々に採点する")
    g.add_argument("--lines", action="store_true", help="1 行を 1 つの案として別々に採点する（言い換えの比較用）")
    sc.add_argument("--target", type=float, default=TARGET)
    sc.add_argument("--json", action="store_true")
    b = sub.add_parser("blocks", help="文書をブロックに分けて採点し、まとまりの下書きを出す")
    b.add_argument("file")
    b.add_argument("--sentences", action="store_true", help="文ごとの P(Yes) も出す")
    b.add_argument("--plan-out", help="まとまりの計画の下書き（JSON）を書き出す")
    b.add_argument("--protect", action="append", help="1 行目がこの正規表現に合う地の文を固定する（何度でも）")
    b.add_argument("--no-default-protect", action="store_true", help="図表のキャプション（図1: …）を固定しない")
    b.add_argument("--style", help="計画に書く文書の文体（例: です・ます調）")
    b.add_argument("--target", type=float, default=TARGET)
    b.add_argument("--json", action="store_true")
    sp = sub.add_parser("split", help="計画どおりにまとまりを切り出し、指示書を書く")
    sp.add_argument("--plan", required=True)
    sp.add_argument("--workdir", required=True)
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--json", action="store_true")
    ck = sub.add_parser("check", help="原文と推敲案の機械チェック")
    ck.add_argument("original")
    ck.add_argument("revised", help="推敲案のファイル。- で標準入力")
    ck.add_argument("--json", action="store_true")
    asm = sub.add_parser("assemble", help="推敲後のまとまりを文書に戻し、報告と記録を書く")
    asm.add_argument("--plan", required=True, help="split が作業ディレクトリに書いた plan.json")
    asm.add_argument("--out", help="推敲後の文書の書き出し先（既定: 原文を上書きする。原文の写しは作業ディレクトリの source.md に残る）")
    asm.add_argument("--force", action="store_true", help="原文が split のあとに書き換えられていても上書きする")
    asm.add_argument("--json", action="store_true", help="記録（record.json と同じ内容）を出す")
    v = sub.add_parser("verify", help="推敲後の文書が記録した本文のままかを確かめる")
    v.add_argument("--workdir", required=True)
    v.add_argument("--final", help="確かめる文書（既定: 記録した書き出し先）")
    v.add_argument("--json", action="store_true")
    a = ap.parse_args()
    return {
        "doctor": cmd_doctor, "setup": cmd_setup, "server": cmd_server, "score": cmd_score, "blocks": cmd_blocks,
        "split": cmd_split, "check": cmd_check, "assemble": cmd_assemble, "verify": cmd_verify,
    }[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
