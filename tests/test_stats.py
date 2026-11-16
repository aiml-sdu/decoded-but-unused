"""Bootstrap / permutation reproducibility."""

import numpy as np

from shingeki.stats.bootstrap import paired_bootstrap_ci
from shingeki.stats.effect_sizes import cohens_dz, fraction_positive
from shingeki.stats.permutation import paired_permutation_p


def test_bootstrap_seeded_reproducible():
    diffs = np.array([1.2, -0.4, 0.8, 0.1, 1.6, 2.0, -0.7])
    a = paired_bootstrap_ci(diffs, n_resamples=2000, seed=42)
    b = paired_bootstrap_ci(diffs, n_resamples=2000, seed=42)
    assert a == b
    c = paired_bootstrap_ci(diffs, n_resamples=2000, seed=43)
    assert a != c


def test_permutation_p_in_unit_interval():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 30)
    b = a + rng.normal(0, 0.1, 30)  # almost identical
    p = paired_permutation_p(a, b, n_perms=2000, seed=0)
    assert 0.0 < p <= 1.0
    # large signal should have small p
    a2 = rng.normal(0, 1, 30)
    b2 = a2 + 5.0
    p2 = paired_permutation_p(a2, b2, n_perms=2000, seed=0)
    assert p2 < 0.01


def test_effect_sizes():
    diffs = np.array([1.0, 2.0, 3.0, -1.0])
    assert abs(cohens_dz(diffs) - (np.mean(diffs) / np.std(diffs, ddof=1))) < 1e-9
    assert fraction_positive(diffs) == 0.75
