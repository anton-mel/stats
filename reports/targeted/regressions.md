# Regressions / improvements

| kind | engine | platform | artifact | workload | program | mode | metric | value | Δ | threshold | commit/version |
|---|---|---|---|---|---|---|---|---|---|---|---|
| regression | pie | m5-max-48g | muse-glimmer-30b-mlx4 | ctx-8k-256 | text-completion-bench | tp1 | decode_tok_s | 26.66 | -14.3% | ±7.1% | c371273ba9cd881d2f7f044f46f83d897b661c3d |
