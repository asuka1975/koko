# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 asuka1975
"""koko のテスト（uv run python -m unittest discover -s tests）。

- 文の取り出し: 判定モデルの学習データを作った前処理と、同じ文を返すか（fixtures/golden_sentences.json）
- 文書の扱い: ブロック分け、計画の検証、組み立て、記録と照合（仮の採点 KOKO_FAKE_SCORER=1 で、GPU なしで動く）
- 採点: 同梱のモデルが、モデルを作った環境で採点した値と同じ値を返すか（fixtures/golden_scores.json）。
  採点サーバの環境（uv run <koko.py> setup）がなければ飛ばす。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".apm" / "skills" / "koko" / "scripts"
KOKO = SCRIPTS / "koko.py"
FIX = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(SCRIPTS))

import kokotext  # noqa: E402


UV = shutil.which("uv")


def koko(*args, fake=True, check=True, cwd=None, input=None):
    """koko を、使う人と同じ `uv run <koko.py>` で動かす。"""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
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
