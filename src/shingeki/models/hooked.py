"""Unified hook-based model wrapper.

Abstracts over two backends:

  - **TransformerLens** for `gpt2`, `gemma-2-2b-it` (TL ships native support).
  - **Custom HF wrapper** for `gemma-3-*`, `qwen2.5-*` (TL 2.16 lacks support).

Both backends expose the same public surface:

  - `cfg.n_layers`, `cfg.d_model`, `cfg.n_heads`, `cfg.d_head`
  - `tokenizer`
  - `device`
  - `W_U`  - unembedding matrix (vocab_size, d_model)  [TL convention]
  - `forward(input_ids) -> logits[B, T, V]`
  - `run_with_cache(input_ids, names: list[str]) -> (logits, dict[str, Tensor])`
  - `run_with_hooks(input_ids, fwd_hooks: list[(name, fn)]) -> logits`

`name` is a TL-style hook string. The supported names are:

  - ``blocks.{l}.hook_resid_pre``      input to layer l (shape [B, T, D])
  - ``blocks.{l}.hook_resid_post``     output of layer l (shape [B, T, D])
  - ``blocks.{l}.hook_attn_out``       residual contribution from attention sublayer
  - ``blocks.{l}.hook_mlp_out``        residual contribution from MLP sublayer
  - ``blocks.{l}.attn.hook_pattern``   attention probabilities [B, H, T, T]

For patching, hook fns receive the named tensor and return either the same
tensor (no-op) or a replacement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
import torch.nn as nn

HookFn = Callable[[torch.Tensor, str], torch.Tensor]
NamedHook = tuple[str, HookFn]


HOOK_RE = re.compile(
    r"^blocks\.(?P<L>\d+)\.(?P<sub>hook_resid_pre|hook_resid_post|hook_resid_mid|"
    r"hook_attn_out|hook_mlp_out|attn\.hook_pattern|attn\.hook_z)$"
)


@dataclass
class ModelCfg:
    n_layers: int
    d_model: int
    n_heads: int
    d_head: int
    vocab_size: int


def parse_hook_name(name: str) -> tuple[int, str]:
    m = HOOK_RE.match(name)
    if not m:
        raise ValueError(f"unrecognised hook name: {name!r}")
    return int(m.group("L")), m.group("sub")


def _find_layers(hf_model: nn.Module) -> nn.ModuleList:
    """Find the *language* decoder-layer ModuleList in an HF causal-LM model.

    Probes a fixed set of common paths in priority order. The order matters for
    multimodal wrappers (Gemma-3 multimodal also has a vision-tower
    `ModuleList` of `SiglipEncoderLayer`s; we must not accidentally pick that).
    """
    candidates = [
        "model.language_model.layers",  # gemma-3 multimodal (Gemma3ForConditionalGeneration)
        "language_model.model.layers",  # other multimodal wrappers
        "language_model.layers",
        "model.model.layers",  # some multimodal text-only stacks
        "model.layers",  # llama / qwen / gemma2 / gemma3-text-only
        "transformer.h",  # gpt2-style
    ]
    for path in candidates:
        obj: nn.Module = hf_model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, nn.ModuleList) and len(obj) > 0:
                # Reject vision-tower stacks: their layer types contain "Vision",
                # "Vit", or "Siglip"; language stacks are "DecoderLayer"-style.
                first_name = type(obj[0]).__name__
                if any(s in first_name for s in ("Vision", "Vit", "Siglip", "Clip")):
                    continue
                return obj
        except AttributeError:
            continue
    raise RuntimeError(
        "Could not locate decoder layers; checked paths: " + ", ".join(candidates)
    )


def _get_lm_head(hf_model: nn.Module) -> nn.Linear:
    for path in ("lm_head", "model.lm_head", "language_model.lm_head"):
        obj: nn.Module = hf_model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, nn.Linear) or hasattr(obj, "weight"):
                return obj  # type: ignore[return-value]
        except AttributeError:
            continue
    raise RuntimeError("Could not locate lm_head")


class HookedModel:
    """Backend-agnostic hooked-model wrapper."""

    def __init__(
        self,
        backend: str,
        model: nn.Module,
        tokenizer,
        cfg: ModelCfg,
        device: str,
        layers: nn.ModuleList | None = None,
    ):
        self.backend = backend
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self._layers = layers  # only used by hf backend
        self._tl = None  # set by tl backend

    # ------------------------------------------------------------ properties

    @property
    def W_U(self) -> torch.Tensor:
        """Unembedding matrix in TL convention: shape [d_model, vocab_size].

        TL: returns its native W_U directly.
        HF: returns ``lm_head.weight.T`` (HF stores as [vocab, d_model])."""
        if self.backend == "tl":
            return self._tl.W_U  # type: ignore[union-attr]
        head = _get_lm_head(self.model)
        return head.weight.detach().T

    # ------------------------------------------------------------- forward

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.backend == "tl":
            return self._tl(input_ids)  # type: ignore[misc]
        kwargs: dict = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        out = self.model(input_ids=input_ids, **kwargs)
        return out.logits if hasattr(out, "logits") else out

    @torch.inference_mode()
    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.forward(input_ids, attention_mask)

    # ------------------------------------------------------- run_with_cache

    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        names: Sequence[str],
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.backend == "tl":
            logits, tl_cache = self._tl.run_with_cache(input_ids, names_filter=list(names))  # type: ignore[union-attr]
            return logits, {n: tl_cache[n] for n in names if n in tl_cache}

        cache: dict[str, torch.Tensor] = {}
        handles = self._register_caching_hooks(names, cache)
        try:
            logits = self.forward(input_ids, attention_mask)
        finally:
            for h in handles:
                h.remove()
        return logits, cache

    # ---------------------------------------------------- run_with_hooks (patch)

    def run_with_hooks(
        self,
        input_ids: torch.Tensor,
        fwd_hooks: Sequence[NamedHook],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.backend == "tl":
            tl_hooks = [(name, self._tl_hook_adapter(fn)) for name, fn in fwd_hooks]
            return self._tl.run_with_hooks(input_ids, fwd_hooks=tl_hooks, return_type="logits")  # type: ignore[union-attr]

        handles = self._register_patching_hooks(fwd_hooks)
        try:
            logits = self.forward(input_ids, attention_mask)
        finally:
            for h in handles:
                h.remove()
        return logits

    @staticmethod
    def _tl_hook_adapter(fn: HookFn):
        # TL hook fn signature is (tensor, hook); ours is (tensor, name).
        def adapt(tensor, hook):
            return fn(tensor, hook.name)

        return adapt

    # ----------------------------------------------- HF hook implementations

    def _register_caching_hooks(
        self, names: Iterable[str], cache: dict[str, torch.Tensor]
    ) -> list[torch.utils.hooks.RemovableHandle]:
        handles: list = []
        names = list(names)
        # Group by layer for efficiency.
        per_layer: dict[int, list[str]] = {}
        for n in names:
            l, _sub = parse_hook_name(n)
            per_layer.setdefault(l, []).append(n)
        layers = self._layers
        assert layers is not None
        for l, sub_names in per_layer.items():
            block = layers[l]
            handles.extend(self._hook_block(block, l, sub_names, cache))
        return handles

    def _hook_block(
        self,
        block: nn.Module,
        l: int,
        names: list[str],
        cache: dict[str, torch.Tensor],
    ) -> list:
        handles = []
        wants = {n.split(f"blocks.{l}.", 1)[-1]: n for n in names}

        if "hook_resid_pre" in wants:
            key = wants["hook_resid_pre"]

            def pre(_module, inputs, _key=key):
                t = inputs[0]
                cache[_key] = t.detach()
                return None

            handles.append(block.register_forward_pre_hook(pre))

        if "hook_resid_post" in wants:
            key = wants["hook_resid_post"]

            def post(_module, _inp, output, _key=key):
                t = output[0] if isinstance(output, tuple) else output
                cache[_key] = t.detach()
                return None

            handles.append(block.register_forward_hook(post))

        if "hook_attn_out" in wants and hasattr(block, "self_attn"):
            key = wants["hook_attn_out"]

            def attn_out(_module, _inp, output, _key=key):
                t = output[0] if isinstance(output, tuple) else output
                cache[_key] = t.detach()
                return None

            handles.append(block.self_attn.register_forward_hook(attn_out))

        if "hook_mlp_out" in wants and hasattr(block, "mlp"):
            key = wants["hook_mlp_out"]

            def mlp_out(_module, _inp, output, _key=key):
                t = output[0] if isinstance(output, tuple) else output
                cache[_key] = t.detach()
                return None

            handles.append(block.mlp.register_forward_hook(mlp_out))

        return handles

    def _register_patching_hooks(
        self, fwd_hooks: Sequence[NamedHook]
    ) -> list:
        handles: list = []
        layers = self._layers
        assert layers is not None
        for name, fn in fwd_hooks:
            l, sub = parse_hook_name(name)
            block = layers[l]
            if sub == "hook_resid_pre":
                def pre(_m, inputs, _name=name, _fn=fn):
                    new_t = _fn(inputs[0], _name)
                    if new_t is None:
                        return None
                    return (new_t,) + tuple(inputs[1:])

                handles.append(block.register_forward_pre_hook(pre))
            elif sub == "hook_resid_post":
                def post(_m, _i, output, _name=name, _fn=fn):
                    if isinstance(output, tuple):
                        new_t = _fn(output[0], _name)
                        if new_t is None:
                            return output
                        return (new_t,) + tuple(output[1:])
                    new_t = _fn(output, _name)
                    return new_t if new_t is not None else output

                handles.append(block.register_forward_hook(post))
            elif sub == "hook_attn_out":
                def aout(_m, _i, output, _name=name, _fn=fn):
                    if isinstance(output, tuple):
                        new_t = _fn(output[0], _name)
                        if new_t is None:
                            return output
                        return (new_t,) + tuple(output[1:])
                    new_t = _fn(output, _name)
                    return new_t if new_t is not None else output

                handles.append(block.self_attn.register_forward_hook(aout))
            elif sub == "hook_mlp_out":
                def mout(_m, _i, output, _name=name, _fn=fn):
                    if isinstance(output, tuple):
                        new_t = _fn(output[0], _name)
                        if new_t is None:
                            return output
                        return (new_t,) + tuple(output[1:])
                    new_t = _fn(output, _name)
                    return new_t if new_t is not None else output

                handles.append(block.mlp.register_forward_hook(mout))
            else:
                raise NotImplementedError(f"hook for {sub!r} not implemented yet")
        return handles
