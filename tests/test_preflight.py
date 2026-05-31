from __future__ import annotations

from pathlib import Path

import pytest

from anvil.preflight import (
    CheckResult,
    PreflightError,
    PreflightReport,
    assert_passed,
    check_compute_capability,
    check_cuda_available,
    check_disk_space,
    check_env_var,
    check_env_vars,
    check_module_importable,
    check_modules_importable,
    check_package_version_below,
    format_report,
    full_run_checks,
    run_preflight,
)


def test_check_env_var_passes_when_set() -> None:
    result = check_env_var("FOO", {"FOO": "bar"})

    assert result.passed is True
    assert result.name == "env:FOO"


def test_check_env_var_fails_when_missing() -> None:
    result = check_env_var("MISSING", {})

    assert result.passed is False
    assert "missing" in result.detail


def test_check_env_var_fails_when_empty() -> None:
    result = check_env_var("FOO", {"FOO": ""})

    assert result.passed is False


def test_check_env_var_fails_when_blank() -> None:
    result = check_env_var("FOO", {"FOO": "   "})

    assert result.passed is False


def test_check_env_vars_returns_one_result_per_name() -> None:
    results = check_env_vars(["A", "B", "C"], {"A": "1", "C": "3"})

    assert [r.name for r in results] == ["env:A", "env:B", "env:C"]
    assert [r.passed for r in results] == [True, False, True]


def test_check_disk_space_passes_for_existing_path(tmp_path: Path) -> None:
    result = check_disk_space(tmp_path, min_gb=0.0001)

    assert result.passed is True
    assert "GB free" in result.detail


def test_check_disk_space_falls_back_to_nearest_parent(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    nonexistent = parent / "does" / "not" / "exist"
    result = check_disk_space(nonexistent, min_gb=0.0001)

    assert result.passed is True
    assert result.name == f"disk:{parent}"


def test_check_disk_space_rejects_zero_min_gb(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="min_gb"):
        check_disk_space(tmp_path, min_gb=0)


@pytest.mark.parametrize("min_gb", [float("inf"), float("nan"), True])
def test_check_disk_space_rejects_invalid_min_gb(tmp_path: Path, min_gb: float) -> None:
    with pytest.raises(ValueError, match="min_gb"):
        check_disk_space(tmp_path, min_gb=min_gb)


def test_check_disk_space_reports_failure_when_threshold_too_high(tmp_path: Path) -> None:
    result = check_disk_space(tmp_path, min_gb=10_000_000.0)

    assert result.passed is False
    assert "need" in result.detail


def test_check_module_importable_passes_for_stdlib() -> None:
    result = check_module_importable("json")

    assert result.passed is True


def test_check_module_importable_fails_for_unknown() -> None:
    result = check_module_importable("totally_not_a_real_module_xyz")

    assert result.passed is False
    assert result.name == "import:totally_not_a_real_module_xyz"


def test_check_module_importable_does_not_execute_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path = tmp_path / "explodes_on_import.py"
    module_path.write_text("raise RuntimeError('imported')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    result = check_module_importable("explodes_on_import")

    assert result.passed is True


def test_check_modules_importable_returns_one_per_name() -> None:
    results = check_modules_importable(["json", "totally_fake"])

    assert [r.passed for r in results] == [True, False]


def test_full_run_checks_include_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    import anvil.preflight as preflight

    for name in [
        "OPENAI_API_KEY",
        "HF_TOKEN",
        "WANDB_API_KEY",
        "HF_HOME",
        "UV_CACHE_DIR",
        "UV_LINK_MODE",
        "WANDB_DIR",
        "TMPDIR",
    ]:
        monkeypatch.setenv(name, "x")

    seen_modules: list[str] = []

    def fake_modules(names: list[str]) -> list[CheckResult]:
        seen_modules.extend(names)
        return [CheckResult(name=f"import:{name}", passed=True, detail="ok") for name in names]

    monkeypatch.setattr(
        preflight,
        "check_disk_space",
        lambda path, min_gb: CheckResult(name="disk", passed=True),
    )
    monkeypatch.setattr(preflight, "check_modules_importable", fake_modules)
    monkeypatch.setattr(
        preflight,
        "check_package_version_below",
        lambda name, upper_major, upper_minor: CheckResult(name=f"version:{name}", passed=True),
    )
    monkeypatch.setattr(
        preflight,
        "check_cuda_available",
        lambda: CheckResult(name="cuda", passed=True),
    )
    monkeypatch.setattr(
        preflight,
        "check_compute_capability",
        lambda min_major, min_minor: CheckResult(name="compute_capability", passed=True),
    )

    results = full_run_checks()

    assert "wandb" in seen_modules
    assert any(result.name == "import:wandb" for result in results)


def test_check_package_version_below_passes_for_installed_package() -> None:
    result = check_package_version_below("numpy", upper_major=999, upper_minor=0)

    assert result.passed is True
    assert result.name == "version:numpy"


def test_check_package_version_below_fails_for_missing_package() -> None:
    result = check_package_version_below("totally-not-a-real-dist", upper_major=2, upper_minor=3)

    assert result.passed is False
    assert "not found" in result.detail


def test_check_package_version_below_fails_when_version_is_too_high() -> None:
    result = check_package_version_below("numpy", upper_major=0, upper_minor=1)

    assert result.passed is False
    assert "need <0.1" in result.detail


def test_preflight_report_all_passed_true_when_all_pass() -> None:
    report = run_preflight(
        [
            CheckResult(name="a", passed=True),
            CheckResult(name="b", passed=True),
        ]
    )

    assert report.all_passed is True
    assert report.failures() == []


def test_preflight_report_all_passed_false_when_any_fail() -> None:
    report = run_preflight(
        [
            CheckResult(name="a", passed=True),
            CheckResult(name="b", passed=False, detail="oops"),
        ]
    )

    assert report.all_passed is False
    assert [r.name for r in report.failures()] == ["b"]


def test_assert_passed_silent_on_success() -> None:
    report = run_preflight([CheckResult(name="a", passed=True)])

    assert_passed(report)


def test_assert_passed_raises_with_details() -> None:
    report = run_preflight(
        [
            CheckResult(name="env:HF_TOKEN", passed=False, detail="missing or empty"),
            CheckResult(name="env:OPENAI_API_KEY", passed=True),
        ]
    )

    with pytest.raises(PreflightError) as exc_info:
        assert_passed(report)

    message = str(exc_info.value)
    assert "env:HF_TOKEN" in message
    assert "missing or empty" in message
    # Passing check should not appear in the failure summary.
    assert "OPENAI_API_KEY" not in message


def test_preflight_failure_carries_report() -> None:
    report = run_preflight([CheckResult(name="a", passed=False, detail="x")])

    with pytest.raises(PreflightError) as exc_info:
        assert_passed(report)

    assert exc_info.value.report is report


def test_format_report_marks_ok_and_fail() -> None:
    report = run_preflight(
        [
            CheckResult(name="a", passed=True, detail="set"),
            CheckResult(name="b", passed=False, detail="missing"),
        ]
    )

    rendered = format_report(report)

    assert "[ok] a: set" in rendered
    assert "[FAIL] b: missing" in rendered


def test_check_cuda_available_returns_a_result() -> None:
    result = check_cuda_available()

    assert isinstance(result, CheckResult)
    assert result.name == "cuda"


def test_check_compute_capability_returns_a_result_without_cuda() -> None:
    result = check_compute_capability(min_major=8, min_minor=0)

    assert isinstance(result, CheckResult)
    assert result.name == "compute_capability"


def test_preflight_report_dataclass_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    report = PreflightReport(results=())

    with pytest.raises(FrozenInstanceError):
        report.results = (CheckResult(name="x", passed=True),)  # type: ignore[misc]
