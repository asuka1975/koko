# koko

koko は、AI が書いた文章を自然な文章に推敲するための、Claude Code 向けのスキルおよびサブエージェントです。

この推敲において、[saihai-v1-8b](https://huggingface.co/hotchpotch/saihai-v1-8b) をベースにファインチューニングした意思決定モデルを利用して、文章を自然な形に変換します。

## 必要なもの

| もの | 使うところ |
| --- | --- |
| [Claude Code](https://claude.com/claude-code) | スキルとサブエージェントを動かす |
| [APM](https://github.com/microsoft/apm) | インストール |
| [uv](https://docs.astral.sh/uv/) | Python のスクリプトを動かす。Python 本体と依存（torch など）も uv が用意する |
| NVIDIA の GPU（VRAM 4 GB 以上）推奨 | 意思決定モデル。GPU がなければ CPU 版の torch で動く（速さは未確認） |
| ディスク 9GB | 意思決定モデルが約 2.6 GB（APM は `apm_modules/` と `.claude/` の両方にコピーするので、入れたプロジェクトで約 5 GB）。採点サーバの環境が約 4 GB（uv のキャッシュ） |

koko は Linux で動作することが確認されています。

## インストール

```bash
apm install <owner>/koko#v0.1.0
```

APM によって、`.claude/skills/koko/`（判定モデルを含む）と `.claude/agents/koko-copy-editor.md` がプロジェクトに展開されます。koko の初回の使用の前には、次のコマンドの実行によって実行環境が用意されます。

```bash
uv run .claude/skills/koko/scripts/koko.py doctor   # 1 行目が koko-doctor: ready なら使える
uv run .claude/skills/koko/scripts/koko.py setup    # needs_setup のとき（torch などを数 GB ダウンロードすることがある）
```

判定モデルの重みのリポジトリでの保持には、Git LFS が採用されています（`.gitattributes`）。LFS の実体が取得されなかった場合、重みのファイルはポインタのまま残ります。このような状態においては、`doctor` の結果は `unavailable` となり、その理由も表示されます。

## 使い方

Claude Code における使用に際し、推敲する文書の指定が必要となります。

```text
koko で docs/article.md を校閲して
```

- koko では、入力されたファイルが直接上書きされます。
- 作業ディレクトリの場所には、`.koko/<stem>/` が用いられます。その中には、元のファイルのコピー（`source.md`）、計画（`plan.json`）、報告（`report.md`）と記録（`record.json`）、そしてまとまりごとの指示書と作業メモ（`chunks/`）が保存され、残ります。
- さらに、まとまりについては、その数が 10 を超える場合、始める前に確認されます。

koko は、ほかのスキルやワークフローの一部として呼び出して利用することもできます。koko の入出力のインターフェースは、[references/interface.md](.apm/skills/koko/references/interface.md) において定義されています。

なお、koko は、すべての文を人が書いたように見える文にできるとは限りません。該当する箇所は、省略された内容や言い換えられた内容とともに、報告（`report.md`）において記録されます。

## 開発

koko の開発における手順は、[DEVELOPMENT.md](DEVELOPMENT.md) に記述されています。

## ライセンス

koko は GNU General Public License v3.0 or later（GPL-3.0-or-later）です（[LICENSE](LICENSE)）。利用者のもとに展開される koko のスキルにも、同じ文面のライセンス文書が付属しています（`.apm/skills/koko/LICENSE`）。

- koko の判定モデル（`model/`）は、Apache License 2.0 のモデルの派生物であり、Apache License 2.0 で配布されています（`model/LICENSE-Apache-2.0.txt`、`model/NOTICE.md`）。
