# koko の入出力の取り決め

koko をほかのスキルやワークフローから呼ぶときに頼ってよい取り決めです。ここに書いたこと（コマンド、ファイル、終了コード、JSON の形）は、koko の版の範囲で保ちます。ここにないことには頼らないでください。

コマンドは `uv run <koko.py> <サブコマンド>` で動かします。`<koko.py>` は、このスキルの `scripts/koko.py` です。

P(Yes) は、同梱の判定モデルが 1 文を「人が書いた文（Wikipedia 由来）」と判定する確率です（0〜1。0 に近いほど AI が書いた文と判定されている）。まとまりの値は、まとまり内の文の P(Yes) の平均です。

## 準備の状態: `doctor`

```bash
uv run <koko.py> doctor --json
```

`status` の値と意味:

| status | 意味 |
|---|---|
| `ready` | 使える |
| `needs_setup` | 採点サーバの環境がない。`uv run <koko.py> setup` で用意できる（初回は torch などを数 GB ダウンロードすることがある） |
| `unavailable` | 使えない。同梱のモデルが壊れている・欠けている（`model.problems`）、または uv がない |

`gpu.cuda` が `false` で `gpu.mps` が `true` なら、採点は MPS（Apple Silicon の GPU）で動きます。どちらも `false` なら、採点は CPU で動きます（CPU での速さは確かめていません）。テキストの出力では、1 行目が `koko-doctor: <status>` です。終了コードは `ready` のとき 0、それ以外は 1 です。

## koko スキルに渡せるもの

Skill ツールで `koko` を呼び、依頼に次を書けます。指定のないものは既定で決まります。

| 項目 | 内容 | 既定 |
|---|---|---|
| 原文 | 推敲する Markdown のパス | （必須） |
| 出力 `OUT` | 推敲後の文書の書き出し先。原文と別のパスを指定すると、原文は書き換えない。原文に書き出すとき、split のあとに原文が書き換えられていたら、上書きせずに止まる | 原文（上書き。原文の写しは `WORK/source.md` に残る） |
| 作業ディレクトリ `WORK` | 計画、指示書、作業メモ、報告、記録を置く | 現在のディレクトリの `.koko/<stem>/` |
| 文体 | です・ます調／である調 | 原文に合わせる |
| 固定する段落 | 書き換えてはいけない段落の 1 行目の形（正規表現）。図表のキャプション（`図1: …` など）は既定で固定 | `^(図\|表)\s*\d+\s*[:：.．]` |
| 文書の要点 | 各節の要点や主張など。各まとまりの「伝えたいこと」の根拠にする | なし（本文から読み取る） |
| 全体への補足 | すべてのまとまりで守ってほしいこと（例: 新しい専門用語を足さない） | なし |
| 目標 | まとまり内の文の平均 P(Yes) の目標 | 0.9 |
| 範囲 | 推敲するまとまりの範囲や上限 | 目標未満のすべて |

ほかのスキルから呼ばれた koko は、ユーザーに直接は聞きません。判断が要ることは、返答の「呼び出し元に確かめたい点」にまとめます。推敲後の本文は返答に載せません。書き込むのは `OUT` と `WORK` の中だけです。

## できあがるもの

| ファイル | 内容 |
|---|---|
| `OUT` | 推敲後の文書。まとまりの外（見出し、表、コード、固定した段落など）は 1 文字も変えない |
| `WORK/record.json` | 記録（下記） |
| `WORK/report.md` | 人が読む報告（まとまりごとの前後の値、達成か未達か、機械チェック、作業メモへのリンク） |
| `WORK/scores.json` | まとまりごと・文ごとの P(Yes)（推敲の前と後） |
| `WORK/chunks/cNN.notes.md` | まとまりごとの作業メモ（補完した整合条件、省いた・言い換えた内容、上がらなかった文） |

### `record.json`

```json
{
  "format": "koko-record-v1",
  "koko_version": "0.1.0",
  "model_id": "<モデルの識別子>",
  "source": "<原文のパス>", "source_sha256": "...",
  "out": "<OUT のパス>", "out_sha256": "...",
  "target": 0.9,
  "summary": {"chunks": 5, "revised": 3, "met": 3, "already_met": 0, "mean_before": 0.015, "mean_after": 0.554, "sentences_before": 16, "sentences_after": 16},
  "chunks": [{"id": "c01", "lines": [5, 5], "section": "...", "status": "met", "n_before": 3, "n_after": 3, "mean_before": 0.03, "mean_after": 0.96, "check_errors": [], "check_warnings": []}]
}
```

`chunks[].status` は `met`（推敲して目標に届いた）、`not_met`（推敲したが届かなかった）、`already_met`（推敲前から届いていた）、`unrevised`（推敲していない）のどれかです。

## 本文が記録のままかを確かめる: `verify`

```bash
uv run <koko.py> verify --workdir <WORK> [--final <OUT>] [--json]
```

読むだけで、何も書きません。1 行目が `koko-check: <状態>` で、終了コードは次のとおりです。

| 状態 | 終了コード | 意味 |
|---|---:|---|
| `ok` | 0 | `OUT` が、記録したときの本文のまま |
| `modified` | 1 | 記録のあとに `OUT` が書き換えられた |
| `no_record` | 3 | `WORK/record.json` がない |
| `missing_final` | 4 | `OUT` がない |
| `model_changed` | 5 | 記録したときと、いまの判定モデルが違う |

`ok` は、本文が記録と同じという意味です。目標を達成したかどうかは `record.json` の `summary` と `chunks[].status` で見ます。

記録は、実行漏れや、あとからの書き換えを見つけるためのものです。同じ権限で本文と記録の両方を書き換えられる環境では、改ざんの防止にはなりません。

## 記録なしで値を測る: `blocks --json`

```bash
uv run <koko.py> blocks <文書> --json
```

文書の地の文をブロック（段落、箇条書きなど）ごとに採点します。`mean` は地の文全体の文の平均、`blocks[].mean` はブロックごとの平均です。koko のあとで本文が書き換えられたときも、いまの本文の値を測れます。

## 機械向けの出力

`doctor`、`score`、`blocks`、`split`、`check`、`assemble`、`verify` は `--json` で JSON を出します。
