from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v8_router_dev_evaluator_registers_project_namespace_before_import() -> None:
    source = (
        ROOT / "tools" / "evaluate_v8_oof_margin_router_dev.py"
    ).read_text(encoding="utf-8")

    assert source.index("runtime = _build_runtime(config)") < source.index(
        "from trifusion.signal_preserving_v8_router import"
    )
