# Anvil

Production-grade LLM fine-tuning with LoRA/QLoRA — dataset synthesis, rigorous eval, cost analysis.

> Numbers, charts, and the full impact-led README land in Phase 9 once the
> pipeline produces real measurements. This stub exists so the repo is public
> from day one (per `DECISIONS.md`); the project status table there tracks
> every phase to v1.0.0.

Anvil is the training-side sibling of [Forge](https://github.com/feRpicoral/forge), which serves open LLMs on a single GPU. Together: train + serve open models efficiently in production.

## Status

See [`DECISIONS.md`](DECISIONS.md) for the phased plan, scope, and stack rationale.

## Local development

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
make check
```

`make check` runs Ruff lint, format check, mypy strict, and pytest — the same checks CI runs on every PR.

## License

[MIT](LICENSE).
