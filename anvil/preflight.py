"""Pre-flight checks for the paid GPU run."""

from __future__ import annotations

import importlib.util
import math
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one preflight check."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class PreflightReport:
    """Aggregate of one preflight run."""

    results: tuple[CheckResult, ...]
    all_passed: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "all_passed", all(r.passed for r in self.results))

    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]


class PreflightError(RuntimeError):
    """Raised by `assert_passed` when any check failed."""

    def __init__(self, report: PreflightReport) -> None:
        lines = [f"  - {r.name}: {r.detail}" for r in report.failures()]
        super().__init__("preflight failed:\n" + "\n".join(lines))
        self.report = report


def check_env_var(name: str, env: Mapping[str, str]) -> CheckResult:
    """Check `env[name]` is set and non-empty."""
    value = env.get(name, "")
    if value.strip():
        return CheckResult(name=f"env:{name}", passed=True, detail="set")
    return CheckResult(name=f"env:{name}", passed=False, detail="missing or empty")


def check_env_vars(names: Sequence[str], env: Mapping[str, str]) -> list[CheckResult]:
    return [check_env_var(name, env) for name in names]


def check_disk_space(path: Path, min_gb: float) -> CheckResult:
    """Check the filesystem holding `path` has at least `min_gb` of free space."""
    if isinstance(min_gb, bool) or not isinstance(min_gb, int | float) or not math.isfinite(min_gb):
        raise ValueError("min_gb must be a positive finite number")
    if min_gb <= 0:
        raise ValueError("min_gb must be a positive finite number")
    target = _nearest_existing_path(path)
    usage = shutil.disk_usage(target)
    free_gb = usage.free / (1024**3)
    passed = free_gb >= min_gb
    return CheckResult(
        name=f"disk:{target}",
        passed=passed,
        detail=f"{free_gb:.1f} GB free (need {min_gb:.1f} GB)",
    )


def check_module_importable(name: str) -> CheckResult:
    """Check that a module can be found without executing its top-level code."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError) as exc:
        return CheckResult(name=f"import:{name}", passed=False, detail=str(exc))
    if spec is None:
        return CheckResult(name=f"import:{name}", passed=False, detail="module not found")
    return CheckResult(name=f"import:{name}", passed=True, detail="ok")


def check_modules_importable(names: Sequence[str]) -> list[CheckResult]:
    return [check_module_importable(name) for name in names]


def check_cuda_available() -> CheckResult:
    """Check that `torch.cuda.is_available()` returns True."""
    try:
        import torch
    except ImportError as exc:
        return CheckResult(name="cuda", passed=False, detail=f"torch missing: {exc}")
    if not torch.cuda.is_available():
        return CheckResult(name="cuda", passed=False, detail="torch.cuda.is_available() is False")
    device_count = torch.cuda.device_count()
    return CheckResult(name="cuda", passed=True, detail=f"{device_count} device(s)")


def check_compute_capability(min_major: int, min_minor: int) -> CheckResult:
    """Check device 0 has compute capability >= (min_major, min_minor).

    RTX 4090 is (8, 9); A100 is (8, 0); H100 is (9, 0).
    """
    try:
        import torch
    except ImportError as exc:
        return CheckResult(name="compute_capability", passed=False, detail=f"torch missing: {exc}")
    if not torch.cuda.is_available():
        return CheckResult(
            name="compute_capability",
            passed=False,
            detail="cuda not available",
        )
    major, minor = torch.cuda.get_device_capability(0)
    passed = (major, minor) >= (min_major, min_minor)
    return CheckResult(
        name="compute_capability",
        passed=passed,
        detail=(f"device 0 capability {major}.{minor} (need {min_major}.{min_minor}+)"),
    )


def run_preflight(checks: Iterable[CheckResult]) -> PreflightReport:
    """Bundle a list of `CheckResult`s into a `PreflightReport`."""
    return PreflightReport(results=tuple(checks))


def assert_passed(report: PreflightReport) -> None:
    """Raise `PreflightError` if any check failed."""
    if not report.all_passed:
        raise PreflightError(report)


def format_report(report: PreflightReport) -> str:
    """Render a one-line-per-check summary suitable for stderr."""
    lines: list[str] = []
    for result in report.results:
        marker = "ok" if result.passed else "FAIL"
        lines.append(f"  [{marker}] {result.name}: {result.detail}")
    return "\n".join(lines)


def rehearsal_checks() -> list[CheckResult]:
    """Checks the M1 rehearsal needs: stdlib imports + writable cwd."""
    import os

    checks = [
        check_module_importable("anvil.data.schema"),
        check_module_importable("anvil.eval.metrics"),
        check_disk_space(Path.cwd(), min_gb=1.0),
    ]
    checks.extend(check_env_vars(["PATH"], os.environ))
    return checks


def full_run_checks() -> list[CheckResult]:
    """Checks the paid GPU run needs: tokens, training stack, CUDA + 4090."""
    import os

    checks: list[CheckResult] = []
    checks.extend(check_env_vars(["OPENAI_API_KEY", "HF_TOKEN", "WANDB_API_KEY"], os.environ))
    checks.append(check_disk_space(Path.cwd(), min_gb=20.0))
    checks.extend(check_modules_importable(["torch", "transformers", "trl", "peft", "datasets"]))
    checks.append(check_cuda_available())
    checks.append(check_compute_capability(min_major=8, min_minor=0))
    return checks


def _nearest_existing_path(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            return Path.cwd()
        candidate = parent
    return candidate


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run anvil preflight checks.")
    parser.add_argument(
        "--mode",
        choices=("rehearsal", "full"),
        default="rehearsal",
        help="Which checks to run (M1 smoke vs. paid GPU).",
    )
    args = parser.parse_args(argv)

    checks = rehearsal_checks() if args.mode == "rehearsal" else full_run_checks()
    report = run_preflight(checks)
    sys.stderr.write(format_report(report) + "\n")
    if not report.all_passed:
        sys.stderr.write("preflight: FAIL\n")
        return 1
    sys.stderr.write("preflight: ok\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
