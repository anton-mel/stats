# stats

Local inference benchmarks for [pie](https://github.com/pie-project/pie) against [Ollama](https://github.com/ollama/ollama) on self-hosted Apple Silicon Macs.

Every pie commit (or a manual run) is measured on the Macs listed in `config.json`. Each model runs twice: pie on its own checkpoint, Ollama on the tag it ships for the same model. Both engines get the same prompts through pie's `scripts/bench/common.py` client, and the site shows prefill and decode side by side.

## Layout

- `matrix/` models, Macs, workloads, engines
- `src/pie_evals/node/` runs on the Mac: `pie-evals-node run --job <spec.json>`
- `src/pie_evals/orchestrate/` `pie-evals expand | check | jobs | collect | report | dashboard`
- `store/records/` parquet results, `reports/` rendered markdown
- `.github/workflows/` `pie-eval` measures, `pages` publishes the site

## Quick start

```sh
uv venv && uv pip install -e ".[dev,orchestrate]"
.venv/bin/pytest -q
.venv/bin/pie-evals expand
.venv/bin/pie-evals jobs --tier targeted --pie-commit <sha> --platform m5-max-48g --out jobs
.venv/bin/pie-evals-node run --job jobs/<job>.json --out out/<job> --download
.venv/bin/pie-evals collect --tier targeted out/<job> && .venv/bin/pie-evals dashboard --out site
```

A Mac runner is registered with `infra/mac/setup-runner.sh`. It needs AC power, Low Power Mode off, Rust with `wasm32-wasip2`, uv and Ollama.
