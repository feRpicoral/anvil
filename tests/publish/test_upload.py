from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from anvil.publish.upload import upload_adapter


def _make_api(commit_url: str = "https://huggingface.co/anvil/contracts/commit/abc") -> MagicMock:
    api = MagicMock()
    api.upload_folder.return_value = MagicMock(commit_url=commit_url)
    api.create_repo.return_value = None
    return api


def test_upload_adapter_rejects_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="adapter_dir"):
        upload_adapter(tmp_path / "missing", repo_id="acme/model")


def test_upload_adapter_rejects_bad_repo_id(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"")

    with pytest.raises(ValueError, match="repo_id"):
        upload_adapter(adapter_dir, repo_id="no-slash")


def test_upload_adapter_rejects_path_traversal(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    with pytest.raises(ValueError, match="repo_id"):
        upload_adapter(adapter_dir, repo_id="../outside/model")


@pytest.mark.parametrize(
    "repo_id",
    [
        "acme/model--x",
        "acme/model..x",
        "acme/-model",
        "acme/model-",
        "acme/.model",
        "acme/model.git",
        f"acme/{'x' * 97}",
    ],
)
def test_upload_adapter_rejects_huggingface_invalid_repo_ids(
    tmp_path: Path,
    repo_id: str,
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    with pytest.raises(ValueError, match="repo_id"):
        upload_adapter(adapter_dir, repo_id=repo_id)


def test_upload_adapter_calls_create_repo_then_upload(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    api = _make_api()

    url = upload_adapter(adapter_dir, repo_id="acme/contracts", api=api)

    api.create_repo.assert_called_once_with(repo_id="acme/contracts", exist_ok=True, private=False)
    api.upload_folder.assert_called_once()
    kwargs = api.upload_folder.call_args.kwargs
    assert kwargs["repo_id"] == "acme/contracts"
    assert kwargs["folder_path"] == str(adapter_dir)
    assert url == "https://huggingface.co/anvil/contracts/commit/abc"


def test_upload_adapter_propagates_private_flag(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    api = _make_api()

    upload_adapter(adapter_dir, repo_id="acme/contracts", api=api, private=True)

    api.create_repo.assert_called_once_with(repo_id="acme/contracts", exist_ok=True, private=True)


def test_upload_adapter_handles_string_commit_info(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    api = MagicMock()
    api.upload_folder.return_value = "https://huggingface.co/raw/string/url"

    url = upload_adapter(adapter_dir, repo_id="acme/contracts", api=api)

    assert url == "https://huggingface.co/raw/string/url"


def test_upload_adapter_falls_back_to_repo_url_when_commit_info_lacks_url(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    api = MagicMock()
    api.upload_folder.return_value = MagicMock(spec=[])

    url = upload_adapter(adapter_dir, repo_id="acme/contracts", api=api)

    assert url == "https://huggingface.co/acme/contracts"


def test_upload_adapter_uses_commit_message(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    api = _make_api()

    upload_adapter(
        adapter_dir,
        repo_id="acme/contracts",
        api=api,
        commit_message="Hand-rolled message",
    )

    assert api.upload_folder.call_args.kwargs["commit_message"] == "Hand-rolled message"
