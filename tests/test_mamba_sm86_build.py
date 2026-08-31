from __future__ import annotations

import pytest


def test_mamba_smoke_receipt_requires_a_complete_locked_source_identity() -> None:
    from tools.smoke_mamba import _source_provenance

    expected = {
        "causal_conv1d_commit": "a" * 40,
        "mamba_commit": "b" * 40,
        "causal_conv1d_patch_sha256": "c" * 64,
        "mamba_patch_sha256": "d" * 64,
    }
    assert _source_provenance(**expected) == expected

    with pytest.raises(ValueError, match="all source provenance fields"):
        _source_provenance(
            causal_conv1d_commit="a" * 40,
            mamba_commit=None,
            causal_conv1d_patch_sha256=None,
            mamba_patch_sha256=None,
        )
