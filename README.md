# Decoded but Unused

Code, data and result summaries for

> Phongsakon Mark Konrad, Toygar Tanyel, Serkan Ayvaz. *Decoded but Unused: Instruction Tuning Routes Moral Framing into the Judgment Readout.* arXiv, 2026. An earlier version was presented at the ICML 2026 Workshop on Mechanistic Interpretability (Seoul).

[Project page](https://aiml-sdu.github.io/decoded-but-unused-page/) · [OpenReview](https://openreview.net/forum?id=da9ylT2doy) · arXiv (identifier to follow)

Language models change their moral verdicts when the same event is reframed. We ask where in the network this happens. We compare one matched pair, Gemma-3-4B pretrained (PT) and instruction-tuned (IT), on 100 moral dilemmas from two fictional sources, each written in a sympathetic, a condemning and an abstract framing. The framing label is linearly decodable in both checkpoints, but only the IT checkpoint uses it. Its paired sympathetic-minus-condemning shift is +12.18 logits against +0.04 in PT. At layer 25, the IT framing direction aligns with the Yes/No readout 8.4 times more strongly than in PT. Patching, a ten-head writer set and a rank-4 subspace locate the pathway, and the behavioural shift replicates in ten instruction-tuned models from three families. In this pair and on this dataset, instruction tuning changes how the representation is read out, not whether it exists.

## Contents

```text
data/                         D0, D6, D7 (the 100-triplet testbed), D1, D2 controls, D3, annotations
src/shingeki/                 experiment package (loading, prompts, probes, lens, patching, audits)
scripts/                      core PT/IT sweep, figure and summary helpers
tests/                        CPU tests, plus optional gated-model token checks
paper_results/                small JSON/CSV summaries behind the paper's numbers
run_*.py                      manuscript-scale experiments
```

The primary testbed is D0 + D6 + D7, giving 100 triplets (50 from *Attack on Titan* and 50 from *Game of Thrones*). Each triplet holds the event fixed and varies the framing. The examples are LLM-assisted constructions over fictional events, not crowd-sourced moral judgments. D2 is a name-reduced control, not an anonymity guarantee.

## Install

Python 3.12. Dependencies are pinned in `uv.lock`.

```bash
uv sync --extra dev
```

## Checks without a GPU

```bash
.venv/bin/python scripts/validate_artifact.py
.venv/bin/python -m pytest tests -m "not gpu"
```

The validator reports a warning for residual source-specific terms in D2. That warning is expected.

## Model runs

These need accepted access to the gated Gemma repositories on Hugging Face, a token in `HF_TOKEN` or `HF_API_KEY` (for example in a `.env` file, which is git-ignored), and a CUDA GPU. All checkpoints are loaded through `shingeki.models.loader.load_hooked`. The raw multimodal Gemma-3 entry point silently yields a randomly initialised `lm_head`.

| Paper item | Script | Stored summary |
|---|---|---|
| Table 1, Fig. 4 (behaviour, probes, alignment PT vs IT) | `run_n100.py` (behaviour), `scripts/run_it_pt_fast.py` (probes, lens, geometry, patching sweep) | `paper_results/matched_pt_it/` |
| Fig. 6 (logit lens) | `scripts/run_it_pt_fast.py` | `paper_results/matched_pt_it/` |
| Table 2 (component patching at layer 20) | `run_asymmetric_patching.py` | `paper_results/asymmetric_patching_n100/` |
| Table 3 (rank-4 subspace ablation) | `run_orthogonal_pca_audit.py`, random controls in `run_n100.py` | `paper_results/orthopca_n100_*`, `paper_results/random_subspace_n100/` |
| Fig. 7, Table 4 (writer heads, mean ablation, random-head null) | not included in this release (helpers in `src/shingeki/patching/head.py`) | `paper_results/writer_heads_n100/`, `paper_results/random_head_baseline/` |
| Table 5, Fig. 8 (evaluative and neutral readouts) | not included in this release | `paper_results/readout_axes_n100/` |
| Sentiment null and matched-question control | not included in this release | `paper_results/sentiment_null/`, `paper_results/sentiment_moralq/` |
| Table 6 (cross-family shift and CIs) | `run_n100.py`, `run_qwen_32b.py`, `scripts/compute_cross_family_ci.py` | `paper_results/cross_model_alignment/` |
| Perpetrator-POV analysis | `data/annotations/inverted_triplets_n100.json` | `paper_results/matched_pt_it/exp01_behavior_n100/` |

Fresh outputs go to `runs/`, which is git-ignored.

## Hardware

All experiments ran on one NVIDIA B200 (180 GB) in BF16. Every model up to 32B parameters fits on one device. The sweep that regenerates the headline numbers takes about 3 GPU-hours on a B200, or about 6 on an A100.

## Known limitations of this release

- The residual and head patching helpers write the clean activation at the corrupt prompt's final index. When the clean prompt is longer, that is an earlier clean token rather than the clean final token.
- `run_asymmetric_patching.py` as released loads D0 only. The stored n = 100 summary in `paper_results/asymmetric_patching_n100/` was produced with the full testbed.
- The scripts for the head ablation, the readout audit and the sentiment controls are not in this release. Their result summaries are included.

All three will be addressed in a later release.

## Citation

```bibtex
@misc{konrad2026decoded,
  title         = {Decoded but Unused: Instruction Tuning Routes Moral Framing into the Judgment Readout},
  author        = {Konrad, Phongsakon Mark and Tanyel, Toygar and Ayvaz, Serkan},
  year          = {2026},
  eprint        = {TBA},
  archivePrefix = {arXiv},
  note          = {An earlier version was presented at the ICML 2026 Workshop on Mechanistic Interpretability}
}
```

The arXiv identifier will be added once it is assigned.

## Licence

MIT, see `LICENSE`.
