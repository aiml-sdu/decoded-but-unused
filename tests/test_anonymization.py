"""Anonymization mapping correctness."""

from shingeki.data.curate_d2 import DEFAULT_NAME_MAP, anonymize_text


def test_aot_terms_replaced():
    text = "Eren Yeager activated the Rumbling, killing many Eldians and Marleyans across Paradis."
    out = anonymize_text(text, DEFAULT_NAME_MAP)
    for forbidden in ("Eren", "Yeager", "Rumbling", "Eldians", "Marleyans", "Paradis"):
        assert forbidden not in out, f"leaked {forbidden!r}: {out}"


def test_word_boundary():
    text = "Erwin's plan succeeded; Mikasa watched Levi cut down the Beast Titan."
    out = anonymize_text(text, DEFAULT_NAME_MAP)
    for forbidden in ("Erwin", "Mikasa", "Levi", "Beast Titan"):
        assert forbidden not in out, f"leaked {forbidden!r}: {out}"
