"""End-of-night synthesis: scan all artifacts, emit angle_report.md.

The report walks through verified vs. failed claims, picks the strongest
honest paper angle, names a backup angle, and lists reviewer objections we
have evidence for. Everything is computed from raw `result.json` files; we
never re-run experiments here.

The function is intentionally tolerant of missing files — the orchestrator may
have skipped phases that failed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _find_bundle(runs_root: Path, exp_id: str) -> Path | None:
    """Return the most-recent runs/<run_id>/<exp_id>/ directory."""
    candidates = sorted(
        runs_root.glob(f"*/{exp_id}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _verdict(value: float | None, *, green: float, yellow: float, higher_is_better: bool = True) -> str:
    if value is None:
        return "RED (missing)"
    if higher_is_better:
        if value >= green:
            return "GREEN"
        if value >= yellow:
            return "YELLOW"
        return "RED"
    if value <= green:
        return "GREEN"
    if value <= yellow:
        return "YELLOW"
    return "RED"


def collect_results(runs_root: Path, nightly_dir: Path) -> dict[str, Any]:
    """Walk runs_root for known exp_ids and the nightly_dir for audit results."""
    out: dict[str, Any] = {"runs_root": str(runs_root), "nightly_dir": str(nightly_dir)}

    # behavior
    out["behavior_d0"] = _load_json((_find_bundle(runs_root, "exp01_behavior") or Path()) / "result.json")
    out["behavior_d2"] = _load_json((_find_bundle(runs_root, "exp16_anon_behavior") or Path()) / "result.json")
    out["behavior_d1"] = _load_json((_find_bundle(runs_root, "exp17_style_behavior") or Path()) / "result.json")
    out["robustness"] = _load_json((_find_bundle(runs_root, "exp02_robustness") or Path()) / "result.json")

    # cosine, probe, lens, geometry
    out["cosine"] = _load_json((_find_bundle(runs_root, "exp03_cosine") or Path()) / "result.json")
    out["probe_d0"] = _load_json((_find_bundle(runs_root, "exp04_probe") or Path()) / "result.json")
    out["probe_d1"] = _load_json((_find_bundle(runs_root, "exp17_style_probe") or Path()) / "result.json")
    out["probe_d2"] = _load_json((_find_bundle(runs_root, "exp16_anon_probe") or Path()) / "result.json")
    out["lens_d0"] = _load_json((_find_bundle(runs_root, "exp05_logit_lens") or Path()) / "result.json")
    out["geometry"] = _load_json((_find_bundle(runs_root, "exp06_geometry") or Path()) / "result.json")

    # patching
    out["patch_S_to_C"] = _load_json((_find_bundle(runs_root, "exp07_resid_S_to_C") or Path()) / "result.json")
    out["patch_C_to_S"] = _load_json((_find_bundle(runs_root, "exp07_resid_C_to_S") or Path()) / "result.json")
    out["component"] = _load_json((_find_bundle(runs_root, "exp08_component") or Path()) / "result.json")
    out["token_pos"] = _load_json((_find_bundle(runs_root, "exp09_token_position") or Path()) / "result.json")
    out["attn_source"] = _load_json((_find_bundle(runs_root, "exp10_attn_source") or Path()) / "result.json")
    out["head"] = _load_json((_find_bundle(runs_root, "exp11_head") or Path()) / "result.json")
    out["mlp_neuron"] = _load_json((_find_bundle(runs_root, "exp12_mlp_neuron") or Path()) / "result.json")

    # steering / projection
    out["steer_L28"] = _load_json((_find_bundle(runs_root, "exp13_steering_L28") or Path()) / "result.json")
    out["steer_L33"] = _load_json((_find_bundle(runs_root, "exp13_steering_L33") or Path()) / "result.json")
    out["projection_L28"] = _load_json((_find_bundle(runs_root, "exp14_projection_ablation_L28") or Path()) / "result.json")

    # cross-model
    out["pt_behavior"] = _load_json((_find_bundle(runs_root, "exp20_pt_behavior") or Path()) / "result.json")
    out["pt_probe"] = _load_json((_find_bundle(runs_root, "exp20_pt_probe") or Path()) / "result.json")
    out["pt_lens"] = _load_json((_find_bundle(runs_root, "exp20_pt_lens") or Path()) / "result.json")
    out["gemma2_behavior"] = _load_json((_find_bundle(runs_root, "exp21_gemma2_behavior") or Path()) / "result.json")
    out["gemma2_probe"] = _load_json((_find_bundle(runs_root, "exp21_gemma2_probe") or Path()) / "result.json")
    out["qwen_behavior"] = _load_json((_find_bundle(runs_root, "exp22_qwen_behavior") or Path()) / "result.json")
    out["qwen_probe"] = _load_json((_find_bundle(runs_root, "exp22_qwen_probe") or Path()) / "result.json")

    # MoralChoice
    out["moralchoice"] = _load_json((_find_bundle(runs_root, "exp18_moralchoice") or Path()) / "result.json")

    # nightly audits
    out["audit_transfer"] = _load_json(nightly_dir / "audit_transfer" / "result.json")
    out["audit_random_subspace"] = _load_json(nightly_dir / "audit_random_subspace" / "result.json")
    out["audit_denominator"] = _load_json(nightly_dir / "audit_denominator" / "result.json")
    out["audit_alignment"] = _load_json(nightly_dir / "audit_alignment" / "result.json")
    out["audit_offtarget"] = _load_json(nightly_dir / "audit_offtarget" / "result.json")
    out["manifest"] = _load_json(nightly_dir / "manifest.json")

    return out


def _md_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    out.append("| " + " | ".join(c.ljust(w) for c, w in zip(rows[0], widths)) + " |")
    out.append("| " + " | ".join("-" * w for w in widths) + " |")
    for r in rows[1:]:
        out.append("| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |")
    return "\n".join(out)


def render_report(results: dict[str, Any]) -> str:
    sections: list[str] = []
    sections.append("# Nightly angle report\n")
    if results.get("manifest"):
        m = results["manifest"]
        sections.append(
            f"Run: `{m.get('git_short', '?')}` ({'dirty' if m.get('git_dirty') else 'clean'}) "
            f"on `{m.get('gpu', '')}`, started {m.get('started_at', '?')}.\n"
        )

    # ============================================================
    # 1. Verified headline results
    # ============================================================
    sections.append("## 1. Verified headline results\n")
    rows: list[list[str]] = [["Claim", "Value", "Verdict"]]

    beh = (results.get("behavior_d0") or {}).get("metrics", {}).get("S_vs_C", {})
    if beh:
        rows.append(
            [
                "D0 S-C mean shift",
                f"{beh.get('mean_shift', 0):+.2f} (CI [{beh.get('ci95_lo', 0):+.2f}, {beh.get('ci95_hi', 0):+.2f}], p={beh.get('perm_p', 1):.4f})",
                _verdict(beh.get("perm_p"), green=0.01, yellow=0.05, higher_is_better=False),
            ]
        )
        rows.append(
            [
                "D0 frac sym>cond",
                f"{beh.get('frac_sym_gt_cond', 0):.2f}",
                _verdict(beh.get("frac_sym_gt_cond"), green=0.7, yellow=0.6),
            ]
        )

    probe = results.get("probe_d0") or {}
    if "metrics" in probe:
        rows.append(
            [
                "D0 probe best acc",
                f"{probe['metrics'].get('best_acc', 0):.3f} (L={probe['metrics'].get('best_layer', '?')})",
                _verdict(probe["metrics"].get("best_acc"), green=0.8, yellow=0.6),
            ]
        )

    pt = (results.get("pt_behavior") or {}).get("metrics", {}).get("S_vs_C", {})
    if pt:
        rows.append(
            [
                "PT S-C mean shift",
                f"{pt.get('mean_shift', 0):+.2f} (p={pt.get('perm_p', 1):.4f})",
                "GREEN (small)" if abs(pt.get("mean_shift", 0)) < 1.0 and pt.get("perm_p", 1) > 0.05 else "YELLOW",
            ]
        )

    pt_probe = results.get("pt_probe") or {}
    if "metrics" in pt_probe:
        rows.append(
            [
                "PT probe best acc",
                f"{pt_probe['metrics'].get('best_acc', 0):.3f}",
                _verdict(pt_probe["metrics"].get("best_acc"), green=0.8, yellow=0.6),
            ]
        )

    proj = (results.get("projection_L28") or {}).get("metrics", {})
    if proj:
        rows.append(
            [
                "Projection ablation shrink @ L=28",
                f"{proj.get('shrink_frac', 0):+.3f} (base {proj.get('S_C_baseline', 0):+.2f} -> abl {proj.get('S_C_ablated', 0):+.2f})",
                _verdict(proj.get("shrink_frac"), green=0.5, yellow=0.2),
            ]
        )

    patch_sc = (results.get("patch_S_to_C") or {}).get("metrics", {})
    if patch_sc:
        rows.append(
            [
                "Residual patching best layer S->C",
                f"L={patch_sc.get('best_layer')} mean rec={patch_sc.get('best_layer_recovery', 0):.3f}",
                _verdict(patch_sc.get("best_layer_recovery"), green=0.4, yellow=0.2),
            ]
        )

    sections.append(_md_table(rows) + "\n")

    # ============================================================
    # 2. Audit verdicts
    # ============================================================
    sections.append("## 2. Audit verdicts\n")
    audit_rows: list[list[str]] = [["Audit", "Outcome", "Verdict"]]

    # transfer
    tr = results.get("audit_transfer") or {}
    if "best_per_pair" in tr:
        d0_d1 = tr["best_per_pair"].get("d0->d1", {})
        d0_d2 = tr["best_per_pair"].get("d0->d2", {})
        audit_rows.append(
            [
                "Probe transfer D0->D1",
                f"acc={d0_d1.get('best_accuracy', 0):.3f} (L={d0_d1.get('best_layer', '?')})",
                _verdict(d0_d1.get("best_accuracy"), green=0.7, yellow=0.5),
            ]
        )
        audit_rows.append(
            [
                "Probe transfer D0->D2",
                f"acc={d0_d2.get('best_accuracy', 0):.3f} (L={d0_d2.get('best_layer', '?')})",
                _verdict(d0_d2.get("best_accuracy"), green=0.7, yellow=0.5),
            ]
        )

    # random subspace
    rs = results.get("audit_random_subspace") or {}
    if "true" in rs and "random_per_rank" in rs:
        for name, sub in rs["true"].items():
            r = sub["rank"]
            rand = rs["random_per_rank"].get(f"rank{r}", {})
            true_shrink = sub["shrink_frac"]
            random_p95 = rand.get("p95", 0)
            specific = abs(true_shrink) > random_p95 + 0.05
            audit_rows.append(
                [
                    f"Projection specificity ({name}, rank={r})",
                    f"true={true_shrink:+.3f}, random p95={random_p95:+.3f}",
                    "GREEN" if specific else "RED",
                ]
            )

    # denominator
    den = results.get("audit_denominator") or {}
    if den:
        for name, sub in den.items():
            if "error" in sub:
                continue
            audit_rows.append(
                [
                    f"Patching stable subset ({name})",
                    f"n_stable={sub.get('n_stable')}/{sub.get('n_triplets')}, best stable L={sub.get('best_layer_stable', '?')}, median={sub.get('best_layer_stable_median', 0):.3f}",
                    _verdict(sub.get("best_layer_stable_median"), green=0.4, yellow=0.2),
                ]
            )

    # alignment
    al = results.get("audit_alignment") or {}
    if "frame_cos_late_mean" in al:
        audit_rows.append(
            [
                "PT/IT framing direction late-layer cos",
                f"{al['frame_cos_late_mean']:.3f}",
                "INFO",
            ]
        )
        audit_rows.append(
            [
                "PT vs IT cos(d_frame, d_judge) peak",
                f"PT={al.get('pt_peak_alignment', 0):.3f}, IT={al.get('it_peak_alignment', 0):.3f}",
                "INFO",
            ]
        )

    # off-target
    ot = results.get("audit_offtarget") or {}
    if "abs_gap_shrink" in ot:
        audit_rows.append(
            [
                "Off-target |gap| shrink (neutral MoralChoice)",
                f"{ot['abs_gap_shrink']:+.3f}",
                "GREEN (specific)" if ot["abs_gap_shrink"] < 0.3 else ("YELLOW" if ot["abs_gap_shrink"] < 0.6 else "RED (non-specific)"),
            ]
        )

    sections.append(_md_table(audit_rows) + "\n")

    # ============================================================
    # 3. Best angle
    # ============================================================
    sections.append("## 3. Strongest paper angle (heuristic)\n")
    angle = _decide_best_angle(results)
    sections.append(angle["text"])
    sections.append("\n")

    # 4. Backup
    sections.append("## 4. Backup angle\n")
    sections.append(angle["backup"])
    sections.append("\n")

    # 5. Claims safe to make
    sections.append("## 5. Claims safe to make (given audit results)\n")
    safe = _claims_safe(results)
    for c in safe:
        sections.append(f"- {c}")
    sections.append("\n")

    # 6. Claims to avoid
    sections.append("## 6. Claims to avoid or weaken\n")
    avoid = _claims_avoid(results)
    for c in avoid:
        sections.append(f"- {c}")
    sections.append("\n")

    # 7. Reviewer objections we are now able to answer
    sections.append("## 7. Reviewer objections + evidence\n")
    for q, a in _objection_pairs(results):
        sections.append(f"**Q.** {q}\n\n**A.** {a}\n")

    # 8. Recommendation
    sections.append("## 8. Recommendation\n")
    sections.append(_recommendation(results))

    return "\n".join(sections)


def _decide_best_angle(results: dict) -> dict:
    pt_probe_acc = (results.get("pt_probe") or {}).get("metrics", {}).get("best_acc", 0)
    pt_shift = abs((results.get("pt_behavior") or {}).get("metrics", {}).get("S_vs_C", {}).get("mean_shift", 0))
    it_shift = abs((results.get("behavior_d0") or {}).get("metrics", {}).get("S_vs_C", {}).get("mean_shift", 0))
    proj_shrink = (results.get("projection_L28") or {}).get("metrics", {}).get("shrink_frac", 0)
    patch_rec = (results.get("patch_S_to_C") or {}).get("metrics", {}).get("best_layer_recovery", 0)
    rs = results.get("audit_random_subspace") or {}
    proj_specific = False
    if "true" in rs and "random_per_rank" in rs:
        for name, sub in rs["true"].items():
            rand = rs["random_per_rank"].get(f"rank{sub['rank']}", {})
            if abs(sub["shrink_frac"]) > rand.get("p95", 0) + 0.05:
                proj_specific = True
                break
    ot = (results.get("audit_offtarget") or {}).get("abs_gap_shrink")
    ot_specific = ot is not None and ot < 0.3

    # Decision tree
    if pt_probe_acc > 0.7 and pt_shift < 1.5 and it_shift > 4 and proj_specific and patch_rec > 0.3:
        text = (
            "**A. Instruction tuning wires moral framing into judgment.**\n\n"
            f"Gemma-3-4B-PT decodes framing (probe acc {pt_probe_acc:.2f}) but does not behaviorally use it "
            f"(|S-C| = {pt_shift:.2f}). Gemma-3-4B-IT shows a large behavioral effect (|S-C| = {it_shift:.2f}). "
            f"Late residual-stream patching transfers framing causally (best mean recovery = {patch_rec:.2f}). "
            f"Projection-removal of the framing subspace at L=28 closes the gap by {proj_shrink:+.2f}, "
            f"and is **specific** (true shrink exceeds random same-rank p95). "
            f"Together this supports a 'late residual pathway, instruction-tuning-installed' mechanism."
        )
        backup = (
            "**B. Decodable-before-used + late residual pathway.** If the IT-only behavioral asymmetry is the weakest part "
            "of the audit (e.g. PT shows residual behavioral effect), demote A to a discussion point and lead with the "
            "more compact mechanism story: framing is decodable from layer ~9, but only becomes judgment-relevant late, "
            "and is removable through a low-rank residual subspace."
        )
    elif proj_specific and patch_rec > 0.3:
        text = (
            "**B. Decodable-before-used + causal late residual pathway.**\n\n"
            f"Probes recover framing from layer 9-15, but the logit-lens readout only diverges late. "
            f"Late residual patching transfers the gap (mean recovery {patch_rec:.2f}) and projection-removal "
            f"specifically closes the gap (true shrink {proj_shrink:+.2f}, exceeds random same-rank p95). "
            "Lead with this as the main result; treat PT-vs-IT as a discussion-section observation."
        )
        backup = "**C. Negative result on additive single-vector steering, plus partial causal evidence via subspace removal and patching.**"
    elif patch_rec > 0.3:
        text = (
            "**C. Distributed late residual pathway, with caveats.**\n\n"
            "Patching transfers but projection-removal is not specific to the framing subspace. "
            "Frame this as 'mechanism is distributed, not localizable to a single component or direction', and lean on the "
            "patching + decodability evidence."
        )
        backup = "**D. Behavioral framing is real and decodable; causal mechanism is partial.**"
    else:
        text = (
            "**D. Behavioral framing + decodability story (causal evidence weak).**\n\n"
            "If audits show the projection-removal claim is non-specific and patching is denominator-driven, retreat to a 4-page "
            "short paper claiming behavioral effect + decodability + negative steering result."
        )
        backup = "Fallback to short paper; highlight negative results on additive steering and the IT-vs-PT asymmetry as a small descriptive contribution."

    return {"text": text, "backup": backup}


def _claims_safe(results: dict) -> list[str]:
    out: list[str] = []
    beh = (results.get("behavior_d0") or {}).get("metrics", {}).get("S_vs_C", {})
    if beh and beh.get("perm_p", 1) < 0.01 and beh.get("frac_sym_gt_cond", 0) > 0.65:
        out.append(
            f"Gemma-3-4B-IT shows a paired sympathetic-condemning shift on D0 "
            f"(mean {beh.get('mean_shift', 0):+.2f}, frac>0 = {beh.get('frac_sym_gt_cond', 0):.2f}, p < 0.01)."
        )
    probe = results.get("probe_d0") or {}
    if probe and probe.get("metrics", {}).get("best_acc", 0) > 0.8:
        out.append(
            f"Linear probes recover the three-class framing from late residual states (best acc {probe['metrics']['best_acc']:.2f})."
        )
    tr = results.get("audit_transfer") or {}
    if "best_per_pair" in tr:
        d0d2 = tr["best_per_pair"].get("d0->d2", {}).get("best_accuracy", 0)
        if d0d2 > 0.6:
            out.append(
                f"D0->D2 (anonymized) probe transfer reaches {d0d2:.2f}, evidence the representation is not purely lexical."
            )
    pt_b = (results.get("pt_behavior") or {}).get("metrics", {}).get("S_vs_C", {})
    pt_p = results.get("pt_probe") or {}
    if pt_b and pt_p:
        if abs(pt_b.get("mean_shift", 0)) < 1.5 and pt_p.get("metrics", {}).get("best_acc", 0) > 0.6:
            out.append(
                "Pretrained Gemma-3-4B does not behaviorally exhibit the framing effect, despite still admitting decodable framing structure."
            )
    rs = results.get("audit_random_subspace") or {}
    if "true" in rs and "random_per_rank" in rs:
        for name, sub in rs["true"].items():
            rand = rs["random_per_rank"].get(f"rank{sub['rank']}", {})
            if abs(sub["shrink_frac"]) > rand.get("p95", 0) + 0.05:
                out.append(
                    f"Projection-removal at L=28 ({name}) closes the S-C gap beyond the random same-rank p95 control."
                )
    return out


def _claims_avoid(results: dict) -> list[str]:
    out: list[str] = []
    rs = results.get("audit_random_subspace") or {}
    if "true" in rs:
        any_specific = False
        for name, sub in rs["true"].items():
            rand = rs["random_per_rank"].get(f"rank{sub['rank']}", {})
            if abs(sub["shrink_frac"]) > rand.get("p95", 0) + 0.05:
                any_specific = True
                break
        if not any_specific:
            out.append(
                "Avoid 'projection ablation isolates the framing subspace' — random same-rank controls match the true direction."
            )
    ot = (results.get("audit_offtarget") or {}).get("abs_gap_shrink")
    if ot is not None and ot > 0.5:
        out.append(
            f"Avoid 'projection ablation is specific to framing' — off-target shrink on neutral MoralChoice prompts is {ot:+.2f}."
        )
    den = results.get("audit_denominator") or {}
    for name, sub in den.items():
        if not isinstance(sub, dict) or "error" in sub:
            continue
        if sub.get("n_small", 0) > 0.4 * sub.get("n_triplets", 1):
            out.append(
                f"Be careful with patching recovery on {name}: {sub['n_small']}/{sub['n_triplets']} triplets are denominator-small."
            )
    return out


def _objection_pairs(results: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    out.append(
        (
            "Is the probe just picking up lexical / lore cues?",
            "We report D0->D2 (anonymized) and D0->D1 (style-controlled) transfer accuracy. The transfer matrix at the best layer is in audit_transfer/result.json.",
        )
    )
    out.append(
        (
            "Is projection-removal just destroying Yes/No confidence?",
            "Random same-rank subspace controls (audit_random_subspace) and an off-target check on neutral MoralChoice prompts (audit_offtarget) are reported.",
        )
    )
    out.append(
        (
            "Are patching results robust to denominator-small cases?",
            "audit_denominator splits triplets by |clean - corrupt| < 2 logits and reports stable-only summaries.",
        )
    )
    out.append(
        (
            "Does instruction tuning create the framing direction?",
            "audit_alignment reports cos(PT.d_frame, IT.d_frame) per layer and cos(d_frame, d_judge) for both checkpoints.",
        )
    )
    return out


def _recommendation(results: dict) -> str:
    angle = _decide_best_angle(results)
    rs = results.get("audit_random_subspace") or {}
    proj_specific = False
    if "true" in rs and "random_per_rank" in rs:
        for name, sub in rs["true"].items():
            rand = rs["random_per_rank"].get(f"rank{sub['rank']}", {})
            if abs(sub["shrink_frac"]) > rand.get("p95", 0) + 0.05:
                proj_specific = True
                break
    patch_ok = (results.get("patch_S_to_C") or {}).get("metrics", {}).get("best_layer_recovery", 0) > 0.3
    if proj_specific and patch_ok:
        return (
            "**Recommendation: target the 8-page long-paper version.** The projection-specificity audit and patching audit "
            "both pass, so the causal-section claims are defensible. Lead with angle A or B above, depending on PT-vs-IT clarity."
        )
    return (
        "**Recommendation: write the 4-page short paper.** Audit results do not robustly support the projection-specificity claim or "
        "the patching mechanism. Lead with behavior + probes + late-readout, and report the failed steering / weak causal evidence "
        "honestly as negative results."
    )


def write_angle_report(runs_root: Path, nightly_dir: Path) -> Path:
    nightly_dir.mkdir(parents=True, exist_ok=True)
    results = collect_results(runs_root, nightly_dir)
    md = render_report(results)
    out = nightly_dir / "angle_report.md"
    out.write_text(md)
    (nightly_dir / "angle_report_inputs.json").write_text(json.dumps(results, indent=2, default=str))
    return out
