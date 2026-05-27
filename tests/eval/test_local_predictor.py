from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from anvil.eval.local_predictor import (
    LocalExtractionPredictor,
    _resolve_dtype,
    build_extraction_messages,
)


def test_build_extraction_messages_shape() -> None:
    messages = build_extraction_messages("This Mutual NDA between Acme and Globex...")

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "JSON object" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"].startswith("This Mutual NDA")


def test_constructor_rejects_empty_base_model() -> None:
    with pytest.raises(ValueError, match="base_model"):
        LocalExtractionPredictor(base_model="")


def test_constructor_rejects_zero_max_new_tokens() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        LocalExtractionPredictor(base_model="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=0)


def test_constructor_rejects_unknown_dtype() -> None:
    with pytest.raises(ValueError, match="torch_dtype"):
        LocalExtractionPredictor(base_model="Qwen/Qwen2.5-0.5B-Instruct", torch_dtype="float8")


def test_variant_base_when_no_adapter() -> None:
    predictor = LocalExtractionPredictor(base_model="Qwen/Qwen2.5-0.5B-Instruct")

    assert predictor.variant == "base"


def test_variant_finetuned_when_adapter_set() -> None:
    predictor = LocalExtractionPredictor(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        adapter_path=Path("outputs/smoke/final"),
    )

    assert predictor.variant == "finetuned"


def test_is_loaded_starts_false() -> None:
    predictor = LocalExtractionPredictor(base_model="Qwen/Qwen2.5-0.5B-Instruct")

    assert predictor.is_loaded is False


def test_ensure_loaded_raises_install_hint_when_transformers_missing() -> None:
    predictor = LocalExtractionPredictor(base_model="Qwen/Qwen2.5-0.5B-Instruct")

    with (
        patch.dict("sys.modules", {"transformers": None}),
        pytest.raises(ImportError, match=r"constraints/train\.txt"),
    ):
        predictor._ensure_loaded()


def test_predict_raises_install_hint_when_transformers_missing() -> None:
    predictor = LocalExtractionPredictor(base_model="Qwen/Qwen2.5-0.5B-Instruct")

    with (
        patch.dict("sys.modules", {"transformers": None}),
        pytest.raises(ImportError, match=r"constraints/train\.txt"),
    ):
        asyncio.run(predictor.predict("contract text"))


def test_resolve_dtype_known_names() -> None:
    torch_mock = MagicMock()
    torch_mock.bfloat16 = "bf16-sentinel"
    torch_mock.float16 = "f16-sentinel"
    torch_mock.float32 = "f32-sentinel"

    assert _resolve_dtype("bfloat16", torch_mock) == "bf16-sentinel"
    assert _resolve_dtype("float16", torch_mock) == "f16-sentinel"
    assert _resolve_dtype("float32", torch_mock) == "f32-sentinel"


def test_resolve_dtype_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown torch_dtype"):
        _resolve_dtype("int4", MagicMock())


def test_constructor_does_not_load_model() -> None:
    # We don't have transformers in CI, so this would crash if the constructor
    # eagerly loaded. The lazy-load contract is what keeps the smoke testable.
    predictor = LocalExtractionPredictor(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        adapter_path=Path("outputs/smoke/final"),
        max_new_tokens=512,
        device_map="cpu",
        torch_dtype="float32",
    )

    assert predictor.is_loaded is False
    assert predictor.variant == "finetuned"
