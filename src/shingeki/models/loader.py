"""Model loaders.

`load_hooked(spec_id_or_spec, dtype, device)` returns a `HookedModel`.

The TL path is used for models TL natively supports (gpt2, gemma-2-*).
The HF path covers everything TL doesn't (gemma-3-*, qwen-2.5-*).
"""

from __future__ import annotations

import torch

from shingeki.env import resolve_hf_token
from shingeki.models.hooked import HookedModel, ModelCfg, _find_layers
from shingeki.models.registry import MODELS, ModelSpec, get


def _select_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None or dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return getattr(torch, str(dtype))


def load_hooked(
    spec_id_or_spec: str | ModelSpec,
    *,
    dtype: str | torch.dtype | None = "auto",
    device: str | None = None,
    attn_implementation: str | None = None,
) -> HookedModel:
    """Load a model in `HookedModel` form.

    `attn_implementation`:
      - None (default): backend chooses (usually sdpa/flash).
      - "eager": disables sdpa/flash so `output_attentions=True` materialises the
        attention probabilities. Required for attention-source analysis (Exp 10)
        on the HF backend.
    """
    spec = spec_id_or_spec if isinstance(spec_id_or_spec, ModelSpec) else get(spec_id_or_spec)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = _select_dtype(dtype)
    token = resolve_hf_token()

    if spec.backend == "tl":
        from transformer_lens import HookedTransformer

        # TL doesn't accept hf_token; rely on huggingface_hub login (env-based).
        if token:
            import os as _os

            _os.environ.setdefault("HF_TOKEN", token)
        tl = HookedTransformer.from_pretrained(
            spec.hf_name, device=device, dtype=torch_dtype
        )
        cfg = ModelCfg(
            n_layers=tl.cfg.n_layers,
            d_model=tl.cfg.d_model,
            n_heads=tl.cfg.n_heads,
            d_head=tl.cfg.d_head,
            vocab_size=tl.cfg.d_vocab,
        )
        m = HookedModel("tl", tl, tl.tokenizer, cfg, device)
        m._tl = tl
        return m

    # HF backend
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_kwargs: dict = {}
    mdl_kwargs: dict = {"dtype": torch_dtype}
    if token:
        tok_kwargs["token"] = token
        mdl_kwargs["token"] = token
    if attn_implementation:
        mdl_kwargs["attn_implementation"] = attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name, **tok_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    hf_model = AutoModelForCausalLM.from_pretrained(spec.hf_name, **mdl_kwargs)
    hf_model.eval()
    hf_model.to(device)

    layers = _find_layers(hf_model)
    hf_cfg = hf_model.config
    # Cope with multiple naming conventions
    text_cfg = getattr(hf_cfg, "text_config", hf_cfg)
    n_layers = getattr(text_cfg, "num_hidden_layers", len(layers))
    d_model = getattr(text_cfg, "hidden_size", None) or getattr(text_cfg, "n_embd")
    n_heads = getattr(text_cfg, "num_attention_heads", None) or getattr(text_cfg, "n_head")
    d_head = d_model // n_heads if (n_heads and d_model) else None
    vocab_size = getattr(text_cfg, "vocab_size", tokenizer.vocab_size)

    cfg = ModelCfg(
        n_layers=int(n_layers),
        d_model=int(d_model),
        n_heads=int(n_heads),
        d_head=int(d_head) if d_head else int(d_model // n_heads),
        vocab_size=int(vocab_size),
    )
    return HookedModel("hf", hf_model, tokenizer, cfg, device, layers=layers)
