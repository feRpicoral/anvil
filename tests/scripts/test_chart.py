from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from scripts.chart import (
    load_cost_payload,
    load_loss_history,
    load_validity_and_field_scores,
    run,
)


@pytest.fixture(autouse=True)
def _close_figs() -> Iterator[None]:
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


def _write_eval_comparison(path: Path, *, finetuned_validity: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_cases": 3,
        "variants": [
            {
                "variant": "base",
                "json_validity_rate": 0.33,
                "field_scores": {"parties": 0.5, "term": 0.4},
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0.0,
                "mean_latency_ms": 0.0,
            },
            {
                "variant": "finetuned",
                "json_validity_rate": finetuned_validity,
                "field_scores": {"parties": 0.9, "term": 0.85},
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0.0,
                "mean_latency_ms": 0.0,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_cost_report(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "training_cost": {"total_usd": 56.76},
        "inference_cost": {
            "self_hosted": [
                {"label": "Llama 3.1 8B on A40", "usd_per_1m_tokens": 0.22, "notes": "test"}
            ],
            "api": [
                {"label": "GPT-4o", "usd_per_1m_tokens": 6.25, "notes": "test"},
            ],
            "input_share": 0.5,
            "notes": [],
        },
        "fine_tuned": {"label": "Llama 3.1 8B QLoRA", "usd_per_1m_tokens": 0.22},
        "breakeven": {
            "primary_api_label": "GPT-4o",
            "primary_api_usd_per_1m": 6.25,
            "months_horizon": 12,
            "monthly_volume_m_tokens": 1.6,
            "curve": [
                {"month": 0, "cumulative_finetuned_usd": 56.76, "cumulative_api_usd": 0.0},
                {"month": 6, "cumulative_finetuned_usd": 60.0, "cumulative_api_usd": 60.0},
                {"month": 12, "cumulative_finetuned_usd": 63.0, "cumulative_api_usd": 120.0},
            ],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_loss_history(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"step": 0, "train_loss": 2.0, "val_loss": 2.1},
        {"step": 10, "train_loss": 1.5, "val_loss": 1.6},
        {"step": 20, "train_loss": 1.2, "val_loss": 1.3},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_validity_and_field_scores(tmp_path: Path) -> None:
    eval_path = _write_eval_comparison(tmp_path / "eval.json")

    validity, fields = load_validity_and_field_scores(eval_path)

    assert validity == {"base": pytest.approx(0.33), "finetuned": 1.0}
    assert fields["finetuned"] == {"parties": 0.9, "term": 0.85}


def test_load_cost_payload(tmp_path: Path) -> None:
    cost_path = _write_cost_report(tmp_path / "cost.json")

    comparison, curve, monthly_volume, primary_api_label = load_cost_payload(cost_path)

    assert len(comparison.self_hosted) == 1
    assert len(comparison.api) == 1
    assert len(curve) == 3
    assert monthly_volume == pytest.approx(1.6)
    assert primary_api_label == "GPT-4o"


def test_load_loss_history(tmp_path: Path) -> None:
    path = _write_loss_history(tmp_path / "loss.json")

    history = load_loss_history(path)

    assert len(history) == 3
    assert history[0].step == 0
    assert history[0].val_loss == pytest.approx(2.1)


def test_load_loss_history_rejects_non_array(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"not": "an array"}', encoding="utf-8")

    with pytest.raises(ValueError, match="JSON array"):
        load_loss_history(path)


@pytest.mark.parametrize(
    "payload",
    [
        [{"step": True, "train_loss": 1.0}],
        [{"step": 1.9, "train_loss": 1.0}],
        [{"step": 1, "train_loss": "1.0"}],
    ],
)
def test_load_loss_history_rejects_coerced_values(
    tmp_path: Path, payload: list[dict[str, object]]
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_loss_history(path)


def test_load_validity_and_field_scores_rejects_non_string_variant(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps({"variants": [{"variant": None, "json_validity_rate": 1.0}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="variant"):
        load_validity_and_field_scores(path)


def test_load_cost_payload_rejects_coerced_scenario_cost(tmp_path: Path) -> None:
    path = tmp_path / "cost.json"
    payload = {
        "inference_cost": {
            "self_hosted": [{"label": "local", "usd_per_1m_tokens": "0.22"}],
            "api": [],
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="usd_per_1m_tokens"):
        load_cost_payload(path)


def test_load_cost_payload_requires_monthly_volume_for_breakeven_curve(tmp_path: Path) -> None:
    path = tmp_path / "cost.json"
    payload = {
        "breakeven": {
            "curve": [{"month": 0, "cumulative_finetuned_usd": 10.0, "cumulative_api_usd": 0.0}]
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="monthly_volume_m_tokens"):
        load_cost_payload(path)


def test_run_writes_four_charts_when_loss_history_absent(tmp_path: Path) -> None:
    eval_path = _write_eval_comparison(tmp_path / "eval.json")
    cost_path = _write_cost_report(tmp_path / "cost.json")
    output_dir = tmp_path / "out"
    args = argparse.Namespace(
        eval_comparison=eval_path,
        cost_report=cost_path,
        loss_history=None,
        output_dir=output_dir,
    )

    rc = run(args)

    assert rc == 0
    expected = {
        "json-validity.png",
        "task-metric-comparison.png",
        "cost-per-1m.png",
        "breakeven.png",
    }
    assert {p.name for p in output_dir.iterdir()} == expected


def test_run_writes_five_charts_with_loss_history(tmp_path: Path) -> None:
    eval_path = _write_eval_comparison(tmp_path / "eval.json")
    cost_path = _write_cost_report(tmp_path / "cost.json")
    loss_path = _write_loss_history(tmp_path / "loss.json")
    output_dir = tmp_path / "out"
    args = argparse.Namespace(
        eval_comparison=eval_path,
        cost_report=cost_path,
        loss_history=loss_path,
        output_dir=output_dir,
    )

    rc = run(args)

    assert rc == 0
    assert (output_dir / "training-loss.png").exists()


def test_run_with_no_inputs_succeeds_with_no_files(tmp_path: Path) -> None:
    args = argparse.Namespace(
        eval_comparison=tmp_path / "missing.json",
        cost_report=tmp_path / "missing.json",
        loss_history=None,
        output_dir=tmp_path / "out",
    )

    rc = run(args)

    assert rc == 0
    assert not (tmp_path / "out").exists()


def test_run_skips_validity_chart_when_eval_payload_lacks_variants(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps({"n_cases": 0, "variants": []}), encoding="utf-8")
    output_dir = tmp_path / "out"
    args = argparse.Namespace(
        eval_comparison=eval_path,
        cost_report=tmp_path / "missing.json",
        loss_history=None,
        output_dir=output_dir,
    )

    rc = run(args)

    assert rc == 0
    assert not (output_dir / "json-validity.png").exists()
