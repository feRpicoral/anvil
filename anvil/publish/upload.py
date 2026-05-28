"""Push a LoRA adapter + model card to Hugging Face Hub.

Thin wrapper around `huggingface_hub.HfApi.upload_folder`. The script
(`scripts/publish.py`) renders the model card first and writes a
`README.md` into the adapter directory; this module just gets the folder
to the Hub. The wrapper accepts an injected `api` so tests can run
without a token.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id  # type: ignore[attr-defined]


def upload_adapter(
    adapter_dir: Path,
    repo_id: str,
    *,
    token: str | None = None,
    commit_message: str = "Upload anvil LoRA adapter",
    private: bool = False,
    api: HfApi | None = None,
) -> str:
    """Upload `adapter_dir` to `repo_id`. Returns the commit URL."""
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter_dir not found: {adapter_dir}")
    if not _is_safe_repo_id(repo_id):
        raise ValueError(f"repo_id must match 'owner/name' shape: {repo_id!r}")

    client = api if api is not None else HfApi(token=token)
    client.create_repo(repo_id=repo_id, exist_ok=True, private=private)
    commit_info: Any = client.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo_id,
        commit_message=commit_message,
    )
    url = getattr(commit_info, "commit_url", None)
    if isinstance(url, str):
        return url
    # Older huggingface_hub versions returned a plain commit URL string.
    if isinstance(commit_info, str):
        return commit_info
    return f"https://huggingface.co/{repo_id}"


def _is_safe_repo_id(repo_id: str) -> bool:
    if repo_id.count("/") != 1:
        return False
    try:
        validate_repo_id(repo_id)
    except HFValidationError:
        return False
    return True
