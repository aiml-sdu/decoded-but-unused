"""Hook plumbing smoke test on gpt2 (CPU)."""

import torch


def test_residual_cache_shapes(gpt2_model):
    m = gpt2_model
    ids = m.tokenizer("Hello world", return_tensors="pt")["input_ids"]
    names = [f"blocks.{l}.hook_resid_post" for l in [0, 5, 11]]
    _, cache = m.run_with_cache(ids, names)
    for n in names:
        t = cache[n]
        assert t.shape == (1, ids.shape[1], m.cfg.d_model), f"bad shape for {n}: {t.shape}"


def test_self_patch_identity(gpt2_model):
    m = gpt2_model
    ids = m.tokenizer("The capital of France is", return_tensors="pt")["input_ids"]
    name = "blocks.5.hook_resid_pre"
    _, cache = m.run_with_cache(ids, [name])
    clean_act = cache[name].detach()

    def hook(t, _n):
        return clean_act.clone()

    base = m(ids).float()
    patched = m.run_with_hooks(ids, [(name, hook)]).float()
    assert torch.allclose(base, patched, atol=1e-4), "self-patch should be identity"
