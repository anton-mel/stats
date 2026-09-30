# Regressions / improvements

| kind | engine | platform | artifact | workload | program | mode | metric | value | Δ | threshold | commit/version |
|---|---|---|---|---|---|---|---|---|---|---|---|
| regression | pie | m5-max-48g | gemma-4-26b-a4b-mlx4 | ob-512-200 | text-completion-bench | tp1 | decode_tok_s | 135.7 | -3.9% | ±2.0% | 2195fa8aa93af86cd2d9297d2b8af1cebef14426 |
| regression | pie | m5-max-48g | gemma-4-26b-a4b-mlx4 | ob-story-200 | text-completion-bench | tp1 | decode_tok_s | 144.9 | -2.5% | ±2.0% | 2195fa8aa93af86cd2d9297d2b8af1cebef14426 |
| regression | pie | m5-max-48g | gemma-4-26b-a4b-mlx4 | control-aa | text-completion-bench | tp1 | decode_tok_s | 149.8 | -2.3% | ±2.0% | 2195fa8aa93af86cd2d9297d2b8af1cebef14426 |
