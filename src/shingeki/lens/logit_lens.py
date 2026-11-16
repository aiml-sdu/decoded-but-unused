"""Raw logit lens.

The trajectory ``yes_no_gap[l] = logit_l(" Yes") - logit_l(" No")`` is computed
by passing each layer's residual stream through the model's final RMSNorm and
the unembedding matrix `W_U`.

Caveat: the raw lens is biased compared to
the tuned lens; we therefore phrase results as "raw logit-lens readout
diverges around layer X", never "the model decides at layer X".

`RAW_LENS_CAVEAT` is exported so it can be embedded directly into the
manuscript.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from shingeki.activations.extract import get_residual_stream
from shingeki.data.schemas import FRAMES, FramingTriplet
from shingeki.io.results import ResultBundle
from shingeki.models.hooked import HookedModel
from shingeki.models.tokens import AnswerTokens, resolve_answer_tokens
from shingeki.prompts.builders import build_aot_prompt
from shingeki.prompts.chat import ModelFamily

RAW_LENS_CAVEAT = (
    "Following Belrose et al. (2023), we phrase logit-lens results in "
    "observational terms (\"the readout diverges around layer X\") rather than "
    "decision-attribution terms (\"the model decides at layer X\"), since the "
    "raw logit lens is known to be biased relative to the tuned lens."
)


def _get_final_norm(model: HookedModel):
    if model.backend == "tl":
        # TL exposes final-layer norm under model.ln_final
        return model._tl.ln_final  # type: ignore[union-attr]
    # HF: try common locations
    for path in (
        "model.norm",
        "model.language_model.norm",
        "language_model.norm",
        "language_model.model.norm",
        "transformer.ln_f",
    ):
        obj = model.model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    # last resort: identity
    return torch.nn.Identity()


@torch.inference_mode()
def logit_lens_yes_no_gap(
    model: HookedModel, prompt, ans: AnswerTokens
) -> np.ndarray:
    """Returns a length-`n_layers` array of yes/no logit gaps per layer.

    Operates on the *final-token* residual stream after each layer's
    output (post-block residual)."""
    if not ans.single_token:
        raise ValueError("logit_lens needs single-token answers")
    h = get_residual_stream(model, prompt, position="final", which="post")  # [L, D]
    norm = _get_final_norm(model)
    h = h.to(model.device)
    # Apply the final norm; if Identity, no-op.
    h_norm = norm(h.to(torch.float32)) if isinstance(norm, torch.nn.Identity) else norm(h)
    h_norm = h_norm.to(torch.float32)
    W_U = model.W_U.detach().to(model.device, torch.float32)  # [D, V]
    logits = h_norm @ W_U  # [L, V]
    gap = logits[:, ans.pos_id] - logits[:, ans.neg_id]
    return gap.detach().cpu().numpy()


def logit_lens_bundle(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    fmt: str = "yesno",
    exp_id: str = "exp05_logit_lens",
) -> ResultBundle:
    ans = resolve_answer_tokens(model.tokenizer, fmt)
    L = model.cfg.n_layers
    gaps = {f: np.zeros((len(triplets), L), dtype=np.float32) for f in FRAMES}
    for i, t in enumerate(triplets):
        for f in FRAMES:
            p = build_aot_prompt(t, f, fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
            gaps[f][i] = logit_lens_yes_no_gap(model, p, ans)
    out_metrics = {
        f: {
            "mean_per_layer": gaps[f].mean(axis=0).tolist(),
            "std_per_layer": gaps[f].std(axis=0).tolist(),
        }
        for f in FRAMES
    }
    out_metrics["sym_minus_cond_per_layer"] = (gaps["sympathetic"] - gaps["condemning"]).mean(axis=0).tolist()
    out_metrics["caveat"] = RAW_LENS_CAVEAT
    return ResultBundle(
        exp_id=exp_id,
        config={"fmt": fmt, "n_triplets": len(triplets), "n_layers": L},
        metrics=out_metrics,
        arrays={f: gaps[f] for f in FRAMES},
    )
