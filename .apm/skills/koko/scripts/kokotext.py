# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 asuka1975
"""文の取り出し（標準ライブラリだけ）。判定モデルの学習データと同じ前処理で、Markdown を 1 文ずつにする。

判定モデルは、学習データを作ったときと同じ形の文を入力にしたときに、学習時と同じように振る舞う。
そのため、ここの処理は、koko の開発で学習データを作ったときの次の処理と同じにしてある。

  - 正規化（normalize_text）、文への分割（split_sentences。括弧の中の句点では区切らない）、括弧書きの除去（strip_parentheticals）
  - Markdown の文書から地の文を取り出す処理（extract）と、そこで使う正規表現
  - 前処理の設定値（NFKC、quote_aware_split: true、max_quote_chars: 300、
    strip_parentheticals: true、max_parenthetical_chars: 60）

学習データを作ったときとの違いは 1 つだけ: コードの囲みを空白 1 つに置き換えるときに改行の数を保ち、`line` が元の文書の行番号と
ずれないようにした（文の中身は変わらない）。学習データで除外していた文の理由（excluded_in_training）は出さない。

表の行は、学習データを作ったときと同じく文にしない。表のセルは簡潔に書くところで、表も推敲の対象にして試したところ、
セルの文が冗長になったため（2026-11-03）。
"""
from __future__ import annotations

import re
import unicodedata

UNICODE_NORMALIZE = "NFKC"
QUOTE_AWARE_SPLIT = True
MAX_QUOTE_CHARS = 300
STRIP_PARENTHETICALS = True
MAX_PARENTHETICAL_CHARS = 60

TERMINALS = "。！？"
CLOSERS = "」』）)】"
_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u2028\u2029\ufeff]")
_RE_MULTISPACE = re.compile(r"[ \t\u3000]+")
_RE_SENT_SPLIT = re.compile(r"(?<=[。！？])(?![」』）)】。！？、])")
_RE_PAREN = re.compile(r"[（(][^（()）]*[)）]")
QUOTE_CLOSER_OF = {"「": "」", "『": "』", "（": "）", "(": ")", "【": "】", "〈": "〉", "《": "》"}
_NO_SPLIT_AFTER = set(CLOSERS + TERMINALS + "、")

FENCE = re.compile(r"```.*?```", re.S)
INLINE_CODE = re.compile(r"`[^`\n]*`")
BOLD = re.compile(r"\*\*(.+?)\*\*")
LIST_PREFIX = re.compile(r"^\s*(?:[-*>]|\d+\.)\s*")
LINK = re.compile(r"(?<!!)\[([^\]]+)\]\([^)]*\)")
IMAGE = re.compile(r"^!\[")


def normalize_text(text: str, form: str | None = UNICODE_NORMALIZE) -> str:
    if form:
        text = unicodedata.normalize(form, text)
    return _RE_CONTROL.sub("", text)


def strip_parentheticals(s: str, max_chars: int = MAX_PARENTHETICAL_CHARS) -> str:
    """入れ子でない短い括弧書き（読み、日付、原語の綴りなど）を取り除く。長い括弧書きは残す。"""

    def repl(m: re.Match) -> str:
        return "" if len(m.group(0)) - 2 <= max_chars else m.group(0)

    out = _RE_PAREN.sub(repl, s)
    return _RE_MULTISPACE.sub(" ", out).strip() if out != s else s


def split_sentences(line: str, quote_aware: bool = QUOTE_AWARE_SPLIT, max_quote_chars: int = MAX_QUOTE_CHARS) -> list[str]:
    """1 行を「。」「！」「？」で文に分ける。閉じ括弧・句点・読点の直前では分けない。

    quote_aware では、閉じが max_quote_chars 字以内にある「」『』（）などの中の句点では分けない。
    """
    line = _RE_MULTISPACE.sub(" ", line).strip()
    if not line:
        return []
    if not quote_aware:
        return [s.strip() for s in _RE_SENT_SPLIT.split(line) if s and s.strip()]
    out: list[str] = []
    stack: list[str] = []
    start = 0
    n = len(line)
    for i, ch in enumerate(line):
        if stack and ch == stack[-1]:
            stack.pop()
        elif ch in QUOTE_CLOSER_OF:
            j = line.find(QUOTE_CLOSER_OF[ch], i + 1)
            if j != -1 and j - i <= max_quote_chars:
                stack.append(QUOTE_CLOSER_OF[ch])
        elif ch in TERMINALS and not stack:
            if i + 1 < n and line[i + 1] in _NO_SPLIT_AFTER:
                continue
            out.append(line[start : i + 1])
            start = i + 1
    out.append(line[start:])
    return [s.strip() for s in out if s and s.strip()]


def extract(md: str) -> list[dict]:
    """Markdown から文を取り出す。

    返す各文: text（採点に使う形。短い括弧書きを除いたもの）、shown（表示用）、line（行番号、1 始まり）、
    kind（「段落」か「箇条書き」）。見出し・表・画像の行とコードの囲みは文にしない。
    """
    md = FENCE.sub(lambda m: " " + "\n" * m.group(0).count("\n"), md)
    rows = []
    for line_no, line in enumerate(md.split("\n"), 1):
        line = line.strip()
        if not line or line.startswith(("|", "#")) or IMAGE.match(line):
            continue
        kind = "箇条書き" if LIST_PREFIX.match(line) else "段落"
        line = LIST_PREFIX.sub("", BOLD.sub(r"\1", line))
        line = normalize_text(line)
        for raw in split_sentences(line):
            shown = INLINE_CODE.sub(lambda m: m.group(0)[1:-1], LINK.sub(r"\1", raw)).strip()
            if not shown:
                continue
            text = strip_parentheticals(shown) if STRIP_PARENTHETICALS else shown
            rows.append({"text": text, "shown": shown, "line": line_no, "kind": kind})
    return rows
