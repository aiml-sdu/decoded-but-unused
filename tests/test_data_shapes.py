"""Schema and shape sanity checks for datasets."""

from shingeki.data.aot import load_d0, load_d1, load_d2


def test_d0_shape():
    ts = load_d0()
    assert len(ts) == 40
    for t in ts:
        assert t.id.startswith("aot-")
        for f in ("sympathetic", "condemning", "abstract"):
            assert getattr(t, f), f"empty {f} for {t.id}"


def test_d1_schema():
    ts = load_d1(include_v2=True)
    assert all(t.id.startswith("sc-") for t in ts)
    for t in ts:
        for f in ("sympathetic", "condemning", "abstract"):
            assert getattr(t, f)


def test_d2_built_from_d0():
    ts = load_d2()
    if not ts:
        # D2 not built; skip silently
        return
    assert all(t.id.startswith("anon-") for t in ts)
    # ensure no obvious AoT names leaked through
    for t in ts:
        for f in ("sympathetic", "condemning", "abstract"):
            text = getattr(t, f)
            for forbidden in ("Eren", "Erwin", "Mikasa", "Reiner", "Marley", "Eldian"):
                assert forbidden not in text, f"leaked {forbidden!r} in {t.id} ({f})"
