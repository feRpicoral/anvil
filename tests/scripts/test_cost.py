from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.cost import (
    CostConfig,
    build_report,
    fold_eval_api_cost,
    load_cost_config,
    run,
    write_report,
)


def _eval_comparison_payload(total_cost_usd: float) -> dict[str, object]:
    return {
        "n_cases": 3,
        "variants": [
            {
                "variant": "base",
                "n_cases": 3,
                "json_validity_rate": 0.0,
                "field_scores": {},
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0.0,
                "mean_latency_ms": 0.0,
            },
            {
                "variant": "gpt-4o",
                "n_cases": 3,
                "json_validity_rate": 1.0,
                "field_scores": {},
                "total_input_tokens": 1000,
                "total_output_tokens": 2000,
                "total_cost_usd": total_cost_usd,
                "mean_latency_ms": 0.0,
            },
        ],
    }


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "cost.toml"
    path.write_text(body, encoding="utf-8")
    return path


def _smoke_body(tmp_path: Path) -> str:
    return f"""
output_path = "{tmp_path / "cost.json"}"
api_keys = ["gpt-4o", "claude-sonnet-4-6"]
input_share = 0.5
months_horizon = 12

[training]
gpu_hours = 4.0
gpu_tier_key = "runpod-rtx-4090-community"
synthesis_api_cost_usd = 54.0
eval_api_cost_usd = 0.0

[fine_tuned]
label = "Llama 3.1 8B QLoRA on RTX 4090"
gpu_tier_key = "runpod-rtx-4090-community"
throughput_tps = 1100.0
throughput_source = "Published benchmark, refresh before quoting"
utilization = 0.8
"""


def test_load_cost_config_minimal(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _smoke_body(tmp_path))

    config = load_cost_config(config_path)

    assert config.training_gpu_hours == 4.0
    assert config.fine_tuned_throughput_tps == 1100.0
    assert config.api_keys == ("gpt-4o", "claude-sonnet-4-6")
    assert config.months_horizon == 12
    assert config.eval_comparison_path is None


def test_load_cost_config_rejects_unknown_gpu_tier(tmp_path: Path) -> None:
    body = _smoke_body(tmp_path).replace(
        '"runpod-rtx-4090-community"',
        '"made-up-tier"',
    )
    config_path = _write_config(tmp_path, body)

    with pytest.raises(ValueError, match="gpu_tier_key"):
        load_cost_config(config_path)


def test_load_cost_config_rejects_unknown_api_key(tmp_path: Path) -> None:
    body = _smoke_body(tmp_path).replace(
        'api_keys = ["gpt-4o", "claude-sonnet-4-6"]',
        'api_keys = ["gpt-4o", "wishful-thinking"]',
    )
    config_path = _write_config(tmp_path, body)

    with pytest.raises(ValueError, match="API pricing key"):
        load_cost_config(config_path)


def test_load_cost_config_rejects_empty_api_keys(tmp_path: Path) -> None:
    body = _smoke_body(tmp_path).replace(
        'api_keys = ["gpt-4o", "claude-sonnet-4-6"]',
        "api_keys = []",
    )
    config_path = _write_config(tmp_path, body)

    with pytest.raises(ValueError, match="api_keys"):
        load_cost_config(config_path)


def test_fold_eval_api_cost_sums_variants(tmp_path: Path) -> None:
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(_eval_comparison_payload(total_cost_usd=6.50)), encoding="utf-8")

    total = fold_eval_api_cost(path)

    assert total == pytest.approx(6.50)


def test_build_report_produces_expected_top_level_keys(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _smoke_body(tmp_path))
    config = load_cost_config(config_path)

    report = build_report(config)

    assert set(report.keys()) == {"training_cost", "inference_cost", "fine_tuned", "breakeven"}
    assert set(report["training_cost"].keys()) >= {"total_usd", "gpu_cost_usd"}
    assert set(report["inference_cost"].keys()) == {"self_hosted", "api", "input_share", "notes"}
    assert "monthly_volume_m_tokens" in report["breakeven"]
    assert "curve" in report["breakeven"]


def test_build_report_breakeven_curve_length_matches_horizon(tmp_path: Path) -> None:
    body = _smoke_body(tmp_path).replace("months_horizon = 12", "months_horizon = 6")
    config_path = _write_config(tmp_path, body)
    config = load_cost_config(config_path)

    report = build_report(config)

    assert len(report["breakeven"]["curve"]) == 7


def test_build_report_includes_eval_api_cost_when_provided(tmp_path: Path) -> None:
    eval_path = tmp_path / "comparison.json"
    eval_path.write_text(
        json.dumps(_eval_comparison_payload(total_cost_usd=6.50)), encoding="utf-8"
    )
    # eval_comparison_path is a top-level key — declare it BEFORE any TOML
    # table or it becomes a sub-key of the previous table.
    body = _smoke_body(tmp_path).replace(
        'api_keys = ["gpt-4o", "claude-sonnet-4-6"]',
        f'eval_comparison_path = "{eval_path}"\napi_keys = ["gpt-4o", "claude-sonnet-4-6"]',
    )
    config_path = _write_config(tmp_path, body)
    config = load_cost_config(config_path)

    report = build_report(config)

    # Training-cost eval_api_cost was 0; eval comparison contributed 6.50.
    assert report["training_cost"]["eval_api_cost_usd"] == pytest.approx(6.50)


def test_build_report_breakeven_volume_matches_hand_computed(tmp_path: Path) -> None:
    # Configure cheap fine-tuned + expensive API + known training cost so the
    # breakeven number is computable from the public formula.
    body = """
output_path = "/tmp/ignored.json"
api_keys = ["gpt-4o"]
input_share = 0.5
months_horizon = 12

[training]
gpu_hours = 0.0
gpu_tier_key = "runpod-rtx-4090-community"
synthesis_api_cost_usd = 120.0
eval_api_cost_usd = 0.0

[fine_tuned]
label = "test"
gpu_tier_key = "runpod-rtx-4090-community"
throughput_tps = 100000.0
throughput_source = "synthetic, for test"
utilization = 1.0
"""
    config_path = _write_config(tmp_path, body)
    config = load_cost_config(config_path)
    report = build_report(config)

    # Training total = $120, fine_tuned per-1M ≈ 0, gpt-4o blended = $6.25/1M.
    # Breakeven = (120/12) / (6.25 - 0) = 1.6.
    assert report["breakeven"]["monthly_volume_m_tokens"] == pytest.approx(1.6, abs=1e-3)


def test_run_writes_output_json(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _smoke_body(tmp_path))
    args = argparse.Namespace(config=config_path)

    rc = run(args)

    assert rc == 0
    output = json.loads((tmp_path / "cost.json").read_text(encoding="utf-8"))
    assert "training_cost" in output


def test_write_report_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "deeply" / "nested" / "out.json"

    write_report(nested, {"hello": "world"})

    assert json.loads(nested.read_text(encoding="utf-8")) == {"hello": "world"}


def test_smoke_config_file_loads() -> None:
    config = load_cost_config(Path("configs/cost-smoke.toml"))

    assert isinstance(config, CostConfig)
    assert config.training_gpu_tier_key == "runpod-rtx-4090-community"
    assert config.months_horizon == 12
