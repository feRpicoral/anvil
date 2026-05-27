"""End-to-end smoke pipeline test.

Drives the three scripts via their `run()` entrypoints with a temp output
dir and the in-repo fixture set; verifies 50 records flow through
synth → curate (no-dedup) → split (allow-overlap) and land as
messages JSONL whose row totals match the input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.curate import run as run_curate
from scripts.split import run as run_split
from scripts.synth import run as run_synth

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "tests" / "data" / "fixtures" / "synthesis"


def _write_smoke_config(tmp_path: Path, num_samples: int) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'backend = "fixture"\n'
        f'fixtures_dir = "{FIXTURES_DIR}"\n'
        f"num_samples = {num_samples}\n"
        f'output_dir = "{tmp_path / "smoke"}"\n'
        "seed = 0\n",
        encoding="utf-8",
    )
    return config_path


def test_smoke_pipeline_produces_messages_jsonl(tmp_path: Path) -> None:
    config_path = _write_smoke_config(tmp_path, num_samples=50)
    smoke_dir = tmp_path / "smoke"

    run_synth(argparse.Namespace(config=config_path))

    raw_path = smoke_dir / "raw_synthesis.jsonl"
    curated_path = smoke_dir / "curated.jsonl"
    run_curate(argparse.Namespace(input=raw_path, output=curated_path, no_dedup=True))

    run_split(
        argparse.Namespace(
            input=curated_path,
            output_dir=smoke_dir,
            ratios=[0.8, 0.1, 0.1],
            seed=0,
            allow_overlap=True,
        )
    )

    train_lines = (smoke_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    val_lines = (smoke_dir / "val.jsonl").read_text(encoding="utf-8").splitlines()
    test_lines = (smoke_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()

    assert len(train_lines) + len(val_lines) + len(test_lines) == 50
    assert len(train_lines) > 0
    assert len(val_lines) > 0
    assert len(test_lines) > 0

    sample = json.loads(train_lines[0])
    assert [m["role"] for m in sample["messages"]] == ["system", "user", "assistant"]
    assert "extraction" not in sample
