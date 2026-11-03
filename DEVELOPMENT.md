# koko の開発

`.apm/` は、koko のパッケージの元となるディレクトリです。`.claude/` は、`apm install` によって生成される展開先のディレクトリです。このため、`.claude/` を直接編集することはありません。さらに、`.claude/` は判定モデルの写しを含むために、リポジトリにも収録されていません。

```bash
apm install --target claude                    # .apm/ の変更を .claude/ に展開する
uv run python -m unittest discover -s tests   # テスト（採点サーバの環境がなければ、モデルを使うテストは飛ばす）
```

採点サーバの依存の変更に際しては、ロックファイルを次のコマンドで再生成する必要があります。

```bash
uv lock --script .apm/skills/koko/scripts/serve_cu130.py
uv lock --script .apm/skills/koko/scripts/serve_cpu.py
```

| 場所 | 中身 |
| --- | --- |
| `.apm/skills/koko/SKILL.md` | スキル本体（分割、指示、組み立て、継ぎ目の点検の手順） |
| `.apm/skills/koko/scripts/koko.py` | CLI（標準ライブラリだけ） |
| `.apm/skills/koko/scripts/kokotext.py` | 文の取り出し（モデルの学習データと同じ前処理。標準ライブラリだけ） |
| `.apm/skills/koko/scripts/serve_cu130.py`、`serve_cpu.py` と `.lock` | 採点サーバの入口（uv のスクリプト）と、依存の版を固定したロックファイル |
| `.apm/skills/koko/scripts/koko_server.py`、`kokomodel.py` | 採点サーバと、モデルの読み込み |
| `.apm/skills/koko/model.json` | 判定モデルの取得先（Hugging Face のリポジトリと版）と、各ファイルの大きさ・SHA-256 |
| `.apm/skills/koko/references/interface.md` | ほかのスキルやワークフローから呼ぶときの入出力の取り決め |
| `.apm/agents/koko-copy-editor.agent.md` | まとまりを推敲するサブエージェント |
| `tests/` | テスト。`fixtures/` の正解は、判定モデルを作った環境の前処理と採点から作った |
| `tools/shard_model.py` | 量子化したモデルを、koko が取得する形（shard と manifest）に書き出す |
| `tools/model_pointer.py` | Hugging Face に上げたモデルの取得先（`.apm/skills/koko/model.json`）を書く |

## 判定モデルを更新する

判定モデルは Hugging Face Hub に置き、koko は `model.json` に書いた版を `~/.cache/koko` に取得します。モデルを作り直したら、次の順に更新します。

```bash
uv run tools/shard_model.py <量子化モデルのディレクトリ> <出力先>
hf upload <user>/<name> <出力先> . --repo-type model
python3 tools/model_pointer.py <出力先> --repo <user>/<name> --revision <アップロードしたコミットの SHA>
```
