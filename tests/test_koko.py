# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 asuka1975
"""koko のテスト（uv run python -m unittest discover -s tests）。

- 文の取り出し: 判定モデルの学習データを作った前処理と、同じ文を返すか（fixtures/golden_sentences.json）
- 文書の扱い: ブロック分け、計画の検証、組み立て、記録と照合（仮の採点 KOKO_FAKE_SCORER=1 で、GPU なしで動く）
- 判定モデルの取得: 手元の HTTP サーバから、取得・続きからの取得・SHA-256 の照合ができるか
- 採点: 判定モデルが、モデルを作った環境で採点した値と同じ値を返すか（fixtures/golden_scores.json）。
  判定モデルの取得と採点サーバの環境（uv run <koko.py> setup）がなければ飛ばす。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".apm" / "skills" / "koko" / "scripts"
KOKO = SCRIPTS / "koko.py"
FIX = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(SCRIPTS))

import kokotext  # noqa: E402


UV = shutil.which("uv")


def koko(*args, fake=True, check=True, cwd=None, input=None, env=None):
    """koko を、使う人と同じ `uv run <koko.py>` で動かす。"""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env or {})}
    env.pop("KOKO_FAKE_SCORER", None)
    if fake:
        env["KOKO_FAKE_SCORER"] = "1"
    r = subprocess.run([UV, "run", "--quiet", str(KOKO), *map(str, args)], capture_output=True, text=True, env=env, cwd=cwd, input=input)
    if check and r.returncode != 0:
        raise AssertionError(f"koko {' '.join(map(str, args))} → exit {r.returncode}\n{r.stdout}\n{r.stderr}")
    return r


class TestSentences(unittest.TestCase):
    def test_same_as_training_preprocess(self):
        gold = json.loads((FIX / "golden_sentences.json").read_text(encoding="utf-8"))
        for name, rows in gold.items():
            with self.subTest(doc=name):
                got = kokotext.extract((FIX / f"doc_{name}.md").read_text(encoding="utf-8"))
                self.assertEqual([(s["text"], s["shown"], s["kind"]) for s in got], [(s["text"], s["shown"], s["kind"]) for s in rows])

    def test_line_numbers_after_code_fence(self):
        md = (FIX / "doc_edge.md").read_text(encoding="utf-8")
        lines = md.split("\n")
        for s in kokotext.extract(md):
            self.assertIn(s["shown"][:4], kokotext.normalize_text(lines[s["line"] - 1]).replace("**", ""), s)

    def test_quote_and_parenthetical(self):
        self.assertEqual(kokotext.split_sentences("「a。b」と言った。次です。"), ["「a。b」と言った。", "次です。"])
        self.assertEqual(kokotext.strip_parentheticals("語(よみ)です。"), "語です。")
        long = "語(" + "あ" * 61 + ")です。"
        self.assertEqual(kokotext.strip_parentheticals(long), long)

    def test_tables_are_not_sentences(self):
        # 表は採点も推敲もしない（セルの文が冗長になったため）
        md = "| もの | 使うところ |\n| --- | --- |\n| uv | 動かす |\n\n地の文です。\n"
        self.assertEqual([s["shown"] for s in kokotext.extract(md)], ["地の文です。"])


DOC = """# 題

## 節A

最初の段落です。二つ目の文です。

導入の文です。
- 項目の一つ目です。
- 項目の二つ目です。

図1: 図の説明です。

| 表 | です |
|---|---|

## 節B

[リンク](https://example.com/a)を含む段落です。数値は3件です。
"""


@unittest.skipUnless(UV, "uv がない")
class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.src = self.dir / "doc.md"
        self.src.write_text(DOC, encoding="utf-8")
        self.work = self.dir / "work"

    def tearDown(self):
        self.tmp.cleanup()

    def plan(self):
        koko("blocks", self.src, "--plan-out", self.work / "plan.json", "--style", "です・ます調")
        plan = json.loads((self.work / "plan.json").read_text(encoding="utf-8"))
        for c in plan["chunks"]:
            c["intent"] = f"{c['id']} の内容"
        (self.work / "plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
        return plan

    def test_caption_is_protected_by_default(self):
        out = json.loads(koko("blocks", self.src, "--json").stdout)
        kinds = {b["start"]: b["kind"] for b in out["blocks"]}
        self.assertEqual(kinds[11], "protected")
        self.assertEqual(kinds[13], "table")
        self.assertNotIn(11, [c["lines"][0] for c in out["suggested_chunks"]])
        out2 = json.loads(koko("blocks", self.src, "--json", "--no-default-protect").stdout)
        self.assertEqual({b["start"]: b["kind"] for b in out2["blocks"]}[11], "caption")

    def test_split_rejects_fixed_blocks_and_empty_intent(self):
        plan = self.plan()
        bad = {**plan, "chunks": [{"id": "x1", "lines": [5, 11], "intent": "a"}, {"id": "x2", "lines": [18, 18], "intent": ""}]}
        (self.work / "bad.json").write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
        r = koko("split", "--plan", self.work / "bad.json", "--workdir", self.dir / "w2", check=False)
        self.assertEqual(r.returncode, 2)
        self.assertIn("固定のブロック", r.stderr)
        self.assertIn("intent", r.stderr)

    def test_assemble_and_verify(self):
        plan = self.plan()
        koko("split", "--plan", self.work / "plan.json", "--workdir", self.work)
        brief = (self.work / "chunks" / "c01.brief.md").read_text(encoding="utf-8")
        self.assertIn("文書の流れ", brief)
        self.assertIn(str(KOKO), brief)
        first = plan["chunks"][0]
        (self.work / "chunks" / f"{first['id']}.revised.md").write_text("書き直した段落です。\n二つ目の文です。\n", encoding="utf-8")
        out = self.dir / "out.md"
        rec = json.loads(koko("assemble", "--plan", self.work / "plan.json", "--out", out, "--json").stdout)
        text = out.read_text(encoding="utf-8")
        # まとまりの外（見出し、キャプション、表）は 1 文字も変わらない
        for keep in ("# 題", "## 節A", "図1: 図の説明です。", "| 表 | です |", "## 節B"):
            self.assertIn(keep, text)
        self.assertIn("書き直した段落です。", text)
        self.assertNotIn("最初の段落です。", text)
        self.assertEqual(rec["summary"]["revised"], 1)
        self.assertEqual(rec["chunks"][0]["status"] in ("met", "not_met"), True)
        r = koko("verify", "--workdir", self.work, check=False)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertTrue(r.stdout.startswith("koko-check: ok"))
        out.write_text(text + "\n追記。\n", encoding="utf-8")
        r = koko("verify", "--workdir", self.work, check=False)
        self.assertEqual((r.returncode, r.stdout.splitlines()[0]), (1, "koko-check: modified"))
        r = koko("verify", "--workdir", self.dir / "none", check=False)
        self.assertEqual((r.returncode, r.stdout.splitlines()[0]), (3, "koko-check: no_record"))

    def test_out_can_overwrite_source(self):
        self.plan()
        koko("split", "--plan", self.work / "plan.json", "--workdir", self.work)
        (self.work / "chunks" / "c01.revised.md").write_text("上書きの段落です。\n", encoding="utf-8")
        koko("assemble", "--plan", self.work / "plan.json", "--out", self.src)
        self.assertIn("上書きの段落です。", self.src.read_text(encoding="utf-8"))
        self.assertEqual((self.work / "source.md").read_text(encoding="utf-8"), DOC)
        self.assertEqual(koko("verify", "--workdir", self.work, "--final", self.src, check=False).returncode, 0)

    def test_default_out_overwrites_source_and_guards_edits(self):
        self.plan()
        koko("split", "--plan", self.work / "plan.json", "--workdir", self.work)
        (self.work / "chunks" / "c01.revised.md").write_text("既定で上書きした段落です。\n", encoding="utf-8")
        koko("assemble", "--plan", self.work / "plan.json")  # --out なしは原文を上書きする
        first = self.src.read_text(encoding="utf-8")
        self.assertIn("既定で上書きした段落です。", first)
        self.assertEqual((self.work / "source.md").read_text(encoding="utf-8"), DOC)
        koko("assemble", "--plan", self.work / "plan.json")  # 自分で書いた結果の上には、もう一度書ける
        # split のあとに原文を手で直したら、上書きせずに止まる
        self.src.write_text(first + "\n手で足した段落です。\n", encoding="utf-8")
        r = koko("assemble", "--plan", self.work / "plan.json", check=False)
        self.assertEqual(r.returncode, 2)
        self.assertIn("書き換えられています", r.stderr)
        self.assertIn("手で足した段落です。", self.src.read_text(encoding="utf-8"))
        koko("assemble", "--plan", self.work / "plan.json", "--force")
        self.assertNotIn("手で足した段落です。", self.src.read_text(encoding="utf-8"))


class _ModelFiles(BaseHTTPRequestHandler):
    """Hugging Face の /<repo>/resolve/<revision>/<name> を真似て、ファイルを返す（Range に応じる）。"""

    root: Path
    log: list

    def do_GET(self):
        name = self.path.rsplit("/", 1)[-1]
        p = self.root / name
        if not self.path.startswith("/user/fake/resolve/" + "a" * 40 + "/") or not p.exists():
            self.send_error(404)
            return
        data = p.read_bytes()
        rng = self.headers.get("Range")
        self.log.append((name, rng))
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        self.wfile.write(data[start:])

    def log_message(self, *args):
        pass


@unittest.skipUnless(UV, "uv がない")
class TestModelFetch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.hub, self.cache = d / "hub", d / "cache"
        self.hub.mkdir()
        (self.hub / "config.json").write_text('{"a": 1}')
        (self.hub / "model-00001-of-00001.safetensors").write_bytes(os.urandom(300_000))
        (self.hub / "manifest.json").write_text("{}")
        files = [{"name": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(self.hub.iterdir())]
        self.source = d / "model.json"
        self.write_source(files)
        handler = type("H", (_ModelFiles,), {"root": self.hub, "log": []})
        self.log = handler.log
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.env = {"KOKO_MODEL_SOURCE": str(self.source), "KOKO_CACHE_DIR": str(self.cache),
                    "HF_ENDPOINT": f"http://127.0.0.1:{self.server.server_port}", "KOKO_MODEL": ""}
        self.dir = self.cache / "models" / ("fake-model-" + "0123456789ab")

    def write_source(self, files):
        self.source.write_text(json.dumps({"format": "koko-model-source-v1", "name": "fake-model", "model_id": "0123456789abcdef" * 4,
                                           "repo": "user/fake", "revision": "a" * 40, "files": files}))

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def doctor_model(self):
        return json.loads(koko("doctor", "--json", check=False, env=self.env).stdout)["model"]

    def test_fetch_resume_and_verify(self):
        self.assertFalse(self.doctor_model()["fetched"])
        koko("setup", "--model-only", env=self.env)
        for p in self.hub.iterdir():
            self.assertEqual((self.dir / p.name).read_bytes(), p.read_bytes())
        m = self.doctor_model()
        self.assertTrue(m["ok"] and m["fetched"], m)
        # 途中まで取ったファイル（.part）があれば、続きから取る
        big = "model-00001-of-00001.safetensors"
        (self.dir / big).unlink()
        (self.dir / (big + ".part")).write_bytes((self.hub / big).read_bytes()[:100_000])
        self.log.clear()
        koko("setup", "--model-only", env=self.env)
        self.assertEqual(self.log, [(big, "bytes=100000-")])
        self.assertEqual((self.dir / big).read_bytes(), (self.hub / big).read_bytes())
        self.assertFalse((self.dir / (big + ".part")).exists())

    def test_sha_mismatch_is_rejected(self):
        files = json.loads(self.source.read_text())["files"]
        files[0]["sha256"] = "0" * 64
        self.write_source(files)
        r = koko("setup", "--model-only", check=False, env=self.env)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("SHA-256", r.stderr)
        self.assertFalse((self.dir / files[0]["name"]).exists())
        self.assertFalse((self.dir / (files[0]["name"] + ".part")).exists())


@unittest.skipUnless(UV, "uv がない")
class TestCheck(unittest.TestCase):
    def run_check(self, orig, rev):
        with tempfile.TemporaryDirectory() as d:
            o, v = Path(d) / "o.md", Path(d) / "v.md"
            o.write_text(orig, encoding="utf-8")
            v.write_text(rev, encoding="utf-8")
            r = koko("check", o, v, "--json", check=False)
            return r.returncode, json.loads(r.stdout)

    def test_errors_and_warnings(self):
        code, res = self.run_check("値は3件です（2つ）。[記事](https://example.com/a)を見ました。", "## 見出し\n値はいくつかでした（これは長めの括弧書きで、本文の内容をここに逃がしています）。")
        self.assertEqual(code, 1)
        self.assertTrue(any("見出し" in x for x in res["errors"]))
        self.assertTrue(any("リンク先" in x for x in res["errors"]))
        self.assertTrue(any("数値" in x for x in res["warnings"]))
        self.assertTrue(any("括弧書き" in x for x in res["warnings"]))

    def test_clean(self):
        self.assertEqual(self.run_check("値は3件です。", "件数は3件です。"), (0, {"errors": [], "warnings": []}))


def runtime_ready() -> bool:
    if not UV:
        return False
    r = koko("doctor", "--json", fake=False, check=False)
    try:
        return json.loads(r.stdout)["status"] == "ready"
    except ValueError:
        return False


@unittest.skipUnless(runtime_ready(), "採点サーバの環境がない（uv run <koko.py> setup で用意する）")
class TestModel(unittest.TestCase):
    def test_scores_match_reference(self):
        gold = json.loads((FIX / "golden_scores.json").read_text(encoding="utf-8"))["scores"]
        r = koko("score", "--lines", "--json", "-", fake=False, input="\n".join(s["text"] for s in gold))
        results = json.loads(r.stdout)["results"]
        device = json.loads(koko("server", "status", fake=False).stdout).get("device")
        if device != "cuda":
            # 正解の値は CUDA で作った。MPS や CPU では計算の順序が違い、値がどれだけずれるかは確かめていない
            self.skipTest(f"正解の値は CUDA で作ったもの（いまの採点は {device}）")
        got = [s["p_yes"] for res in results for s in res["sentences"]]
        self.assertEqual(len(got), len(gold))
        for g, p in zip(gold, got):
            self.assertAlmostEqual(p, g["p_yes"], places=6, msg=g["text"])

    def test_doctor_deep(self):
        info = json.loads(koko("doctor", "--deep", "--json", fake=False).stdout)
        self.assertTrue(info["model"]["ok"], info["model"]["problems"])


if __name__ == "__main__":
    unittest.main()
