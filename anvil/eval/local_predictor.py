"""Local HF-backed extraction predictor.

One class for both the `base` and `fine-tuned` eval variants. With
`adapter_path=None` it serves the base model; with a PEFT adapter directory
it loads the adapter on top. Imports are lazy so `anvil` installs without
the training stack; the CUDA-coupled deps come from `constraints/train.txt`.

Install:

    uv pip install -c constraints/train.txt transformers peft accelerate
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anvil.eval.runner import Prediction

_INSTALL_HINT = (
    "Local predictor stack not installed. Run:\n"
    "  uv pip install -c constraints/train.txt transformers peft accelerate"
)

_SYSTEM_PROMPT = (
    "You extract structured contract fields from legal documents. "
    "Return a single JSON object that conforms exactly to the provided schema. "
    "Use null where the contract does not specify a value. "
    "Do not include any text outside the JSON object."
)

_KNOWN_DTYPES = ("bfloat16", "float16", "float32")


def build_extraction_messages(contract_text: str) -> list[dict[str, str]]:
    """Render the chat-format messages an extraction predictor sends."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": contract_text},
    ]


class LocalExtractionPredictor:
    """HF transformers + optional PEFT LoRA adapter predictor."""

    def __init__(
        self,
        *,
        base_model: str,
        adapter_path: Path | None = None,
        max_new_tokens: int = 1024,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
    ) -> None:
        if not base_model:
            raise ValueError("base_model must be non-empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if torch_dtype not in _KNOWN_DTYPES:
            raise ValueError(f"torch_dtype must be one of {_KNOWN_DTYPES}, got {torch_dtype!r}")
        self._base_model_name = base_model
        self._adapter_path = adapter_path
        self._max_new_tokens = max_new_tokens
        self._device_map = device_map
        self._torch_dtype_str = torch_dtype
        self._tokenizer: Any = None
        self._model: Any = None

    @property
    def variant(self) -> str:
        return "finetuned" if self._adapter_path is not None else "base"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self.is_loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        torch_dtype = _resolve_dtype(self._torch_dtype_str, torch)
        tokenizer = AutoTokenizer.from_pretrained(self._base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self._base_model_name,
            torch_dtype=torch_dtype,
            device_map=self._device_map,
        )
        if self._adapter_path is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(_INSTALL_HINT) from exc
            model = PeftModel.from_pretrained(model, str(self._adapter_path))
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    async def predict(self, contract_text: str) -> Prediction:
        self._ensure_loaded()
        try:
            import torch
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        messages = build_extraction_messages(contract_text)
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_tokens = int(inputs["input_ids"].shape[-1])

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        generated_ids = outputs[0][input_tokens:]
        raw_output = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        return Prediction(
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=int(generated_ids.shape[-1]),
            cost_usd=0.0,
        )


def _resolve_dtype(name: str, torch_module: Any) -> Any:
    if name == "bfloat16":
        return torch_module.bfloat16
    if name == "float16":
        return torch_module.float16
    if name == "float32":
        return torch_module.float32
    raise ValueError(f"unknown torch_dtype: {name!r}")
