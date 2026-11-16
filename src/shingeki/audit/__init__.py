"""Audit modules for the nightly verification run.

Six audits, each addressing a specific risk in the headline claims:
  manifest          : env / git / model SHA / token-id / chat-template snapshot
  transfer_matrix   : cross-domain probe transfer (D0 / D1 / D2)
  random_subspace   : random same-rank subspace controls for projection ablation
  denominator       : split residual-patching results by denominator stability
  direction_alignment: PT vs IT framing/judgment direction alignment
  offtarget         : projection ablation on unrelated MoralChoice prompts
"""
