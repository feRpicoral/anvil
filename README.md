# Anvil

Production-grade LLM fine-tuning with LoRA/QLoRA — dataset synthesis, rigorous eval, cost analysis.

Anvil is the training-side sibling of [Forge](https://github.com/feRpicoral/forge), which serves open LLMs on a single GPU. Together: train + serve open models efficiently in production.

## Local development

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
make check
```

`make check` runs Ruff lint, format check, mypy strict, and pytest — the same checks CI runs on every PR.

## License

[MIT](LICENSE).
