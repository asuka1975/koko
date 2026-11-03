# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 asuka1975
"""判定モデルの読み込みと採点（koko の実行環境の torch で動かす）。

モデルは、koko の開発で saihai-v1-8b（hotchpotch、Apache-2.0）をファインチューニングし、2bit に量子化したもの。
読み込みと入力の組み立ては、学習と量子化のときと同じ処理にしてある（同じ入力に同じ値を返すため）。違いは次の 2 つ:

  - 重みを shard（manifest.json の shards）に分けて持ち、読み込むときにまとめる
  - 読み込む前に、各ファイルの大きさを manifest.json と照らす（Git LFS のポインタのままのファイルを見つける）

P(Yes) は、その 1 文を Wikipedia 由来（人が書いた側）と判定する確率。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

INSTRUCTION = "この文は自然な日本語か？"
CRITERIA = {"false": "No（不自然な日本語）", "true": "Yes（自然な日本語）"}
LINEARS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
_CJK = "ぁ-ヿ一-鿿々〆"
_CJK_SPACE = re.compile(rf"(?<=[{_CJK}]) +(?=[0-9A-Za-z])|(?<=[0-9A-Za-z]) +(?=[{_CJK}])")


class ModelError(RuntimeError):
    pass


def read_manifest(model_dir: Path) -> dict:
    p = Path(model_dir) / "manifest.json"
    if not p.exists():
        raise ModelError(f"manifest.json がありません: {p}")
    return json.loads(p.read_text())


def check_files(model_dir: Path, manifest: dict) -> list[str]:
    """大きさが manifest と合わないファイル（欠け、LFS のポインタなど）の説明を返す。空なら問題なし。"""
    problems = []
    for f in manifest["files"]:
        p = Path(model_dir) / f["name"]
        if not p.exists():
            problems.append(f"{f['name']}: ありません")
        elif p.stat().st_size != f["bytes"]:
            head = p.open("rb").read(64)
            lfs = head.startswith(b"version https://git-lfs")
            problems.append(f"{f['name']}: 大きさが {p.stat().st_size} バイトで、{f['bytes']} バイトのはず" + ("（Git LFS のポインタのまま。`git lfs pull` が必要）" if lfs else ""))
    return problems


def strip_cjk_space(text: str) -> str:
    """和文と数字・英字の間の半角スペースを取る（「1 つ」→「1つ」）。判定モデルは、この形で学習している。"""
    return _CJK_SPACE.sub("", text)


def render(sentence: str, contract: dict) -> str:
    """saihai の noul 型の質問（false、true の順）を、学習時と同じ文字列にする。"""
    codes = contract["codes"]
    payload = {
        "instruction": INSTRUCTION,
        "state": sentence,
        "type": "noul",
        "options": [
            {"code": codes[0], "label": "false", "definition": CRITERIA["false"]},
            {"code": codes[1], "label": "true", "definition": CRITERIA["true"]},
        ],
    }
    return contract["prompt_prefix"] + json.dumps(payload, ensure_ascii=False) + contract["prompt_suffix"]


def _modules():
    import torch
    import torch.nn.functional as F

    def unpack(packed, bits: int):
        shifts = torch.arange(0, 8, bits, dtype=torch.uint8, device=packed.device)
        return ((packed.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).reshape(packed.shape[0], -1)

    def dequantize(packed, scale, zero, bits: int, group: int, dtype=torch.bfloat16):
        codes = unpack(packed, bits)
        r, c = codes.shape
        w = (codes.reshape(r, c // group, group).to(scale.dtype) - zero.unsqueeze(-1)) * scale.unsqueeze(-1)
        return w.reshape(r, c).to(dtype)

    class QuantLinear(torch.nn.Module):
        def __init__(self, qweight, scale, zero, bits: int, group: int):
            super().__init__()
            self.bits, self.group = bits, group
            self.out_features, self.in_features = qweight.shape[0], qweight.shape[1] * 8 // bits
            self.register_buffer("qweight", qweight)
            self.register_buffer("scale", scale)
            self.register_buffer("zero", zero)

        def forward(self, x):
            return F.linear(x, dequantize(self.qweight, self.scale, self.zero, self.bits, self.group).to(x.dtype))

    class QuantEmbedding(torch.nn.Module):
        def __init__(self, qweight, scale, zero, bits: int, group: int):
            super().__init__()
            self.bits, self.group = bits, group
            self.register_buffer("qweight", qweight)
            self.register_buffer("scale", scale)
            self.register_buffer("zero", zero)

        def forward(self, ids):
            flat = ids.reshape(-1)
            rows = dequantize(self.qweight[flat], self.scale[flat], self.zero[flat], self.bits, self.group)
            return rows.reshape(*ids.shape, rows.shape[-1])

    class DecisionModel(torch.nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone, self.head = backbone, head

        def forward(self, input_ids, attention_mask):
            hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
            last = attention_mask.sum(-1) - 1
            state = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
            return self.head(state.float())

    return QuantLinear, QuantEmbedding, DecisionModel


def load_model(model_dir: Path, device: str):
    import torch
    from safetensors.torch import load_file
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel, LlamaRotaryEmbedding

    QuantLinear, QuantEmbedding, DecisionModel = _modules()
    model_dir = Path(model_dir)
    manifest = read_manifest(model_dir)
    problems = check_files(model_dir, manifest)
    if problems:
        raise ModelError("モデルのファイルが壊れているか、欠けています:\n" + "\n".join(f"  - {p}" for p in problems))
    spec = json.loads((model_dir / "quant_config.json").read_text())
    cfg = LlamaConfig.from_pretrained(str(model_dir))
    cfg._attn_implementation = "sdpa"
    cfg.use_cache = False
    t = {}
    for shard in manifest["shards"]:
        t.update(load_file(str(model_dir / shard), device=device))

    def quant(cls, prefix, s):
        return cls(t[f"{prefix}.qweight"], t[f"{prefix}.scale"], t[f"{prefix}.zero"], s["bits"], s["group"])

    with torch.device("meta"):
        model = LlamaModel(cfg)
    model.embed_tokens = quant(QuantEmbedding, "embed_tokens", spec["embedding"])
    for i, layer in enumerate(model.layers):
        for name in LINEARS:
            parent, child = name.split(".")
            setattr(getattr(layer, parent), child, quant(QuantLinear, f"layers.{i}.{name}", spec["linear"]))
        for norm in ("input_layernorm", "post_attention_layernorm"):
            getattr(layer, norm).weight = torch.nn.Parameter(t[f"layers.{i}.{norm}.weight"], requires_grad=False)
    model.norm.weight = torch.nn.Parameter(t["norm.weight"], requires_grad=False)
    model.rotary_emb = LlamaRotaryEmbedding(cfg, device=device)
    head = torch.nn.Linear(cfg.hidden_size, 2, bias="head.bias" in t, device="meta")
    head.weight = torch.nn.Parameter(t["head.weight"].float(), requires_grad=False)
    if "head.bias" in t:
        head.bias = torch.nn.Parameter(t["head.bias"].float(), requires_grad=False)
    left = [n for n, p in [*model.named_parameters(), *model.named_buffers()] if p.is_meta]
    if left:
        raise ModelError(f"重みが足りません: {left[:5]}")
    return DecisionModel(model, head).eval(), manifest, spec


class Scorer:
    """1 文ずつ（バッチ 1 で）採点する。

    まとめて採点すると、同じバッチに入るほかの文の長さで値が最大 0.02 ほど揺れる（2bit の重みを毎回 bf16 に
    戻す計算の、形による丸めの差）。どの呼び出しでも同じ文に同じ値を返すために、1 文ずつにしている。
    """

    def __init__(self, model_dir: Path, device: str):
        import torch
        from tokenizers import Tokenizer

        self.torch = torch
        self.device = device
        self.model, self.manifest, spec = load_model(model_dir, device)
        self.contract = json.loads((Path(model_dir) / "decision-inference.json").read_text())
        self.tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.strip_space = bool(spec.get("strip_space"))

    def ids(self, text: str) -> list[int]:
        if self.strip_space:
            text = strip_cjk_space(text)
        return self.tokenizer.encode(render(text, self.contract), add_special_tokens=False).ids

    def p_yes(self, text: str) -> float:
        torch = self.torch
        ids = torch.tensor([self.ids(text)], dtype=torch.long, device=self.device)
        mask = torch.ones_like(ids)
        with torch.inference_mode():
            return self.model(ids, mask).softmax(-1)[0, 1].float().item()
