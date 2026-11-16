"""Shared pytest fixtures."""

import os
import sys

import pytest

# allow `from shingeki...` to work without install (also via the editable install)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="session")
def gpt2_model():
    from shingeki.models.loader import load_hooked

    return load_hooked("gpt2", dtype="float32", device="cpu")
