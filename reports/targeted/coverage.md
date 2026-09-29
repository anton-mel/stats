# Coverage — targeted

| status | cells |
|---|---|
| pass | 43 |
| fail | 0 |
| declared_unsupported | 45 |
| not_run | 200 |
| noisy | 7 |

## Gaps (expected supported, but not passing)

| engine | platform | artifact | workload | program | mode | status | error | message |
|---|---|---|---|---|---|---|---|---|
| ollama | m5-max-48g | gemma-4-26b-a4b-ollama | ob-story-200 | text-completion-bench | tp1 | noisy |  |  |
| ollama | m5-max-48g | gemma-4-26b-a4b-ollama | ob-512-200 | text-completion-bench | tp1 | noisy |  |  |
| ollama | m5-max-48g | gemma-4-26b-a4b-ollama | ctx-4k-256 | text-completion-bench | tp1 | noisy |  |  |
| ollama | m5-max-48g | gemma-4-26b-a4b-ollama | ctx-8k-256 | text-completion-bench | tp1 | noisy |  |  |
| ollama | m5-max-48g | gemma-4-26b-a4b-ollama-mlx | ob-short-100 | text-completion-bench | tp1 | noisy |  |  |
| ollama | m5-max-48g | gemma-4-26b-a4b-ollama-mlx | cache-2k | text-completion-bench | tp1 | noisy |  |  |
| ollama | m5-max-48g | muse-glimmer-30b-ollama | ob-512-200 | text-completion-bench | tp1 | noisy |  |  |
| pie | m1-max-32g | gemma-4-26b-a4b-mlx4 | control-aa | text-completion-bench | tp1 | not_run |  |  |
| pie | m1-max-32g | gemma-4-26b-a4b-mlx4 | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| pie | m1-max-32g | gemma-4-26b-a4b-mlx4 | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| pie | m1-max-32g | gemma-4-26b-a4b-mlx4 | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| pie | m1-max-32g | gemma-4-26b-a4b-mlx4 | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| pie | m1-max-32g | gemma-4-26b-a4b-mlx4 | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| pie | m1-max-32g | gemma-4-26b-a4b-mlx4 | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| pie | m1-max-32g | gemma-4-26b-a4b-mlx4 | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| pie | m1-max-32g | gemma-4-26b-a4b-mlx4 | c4 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | gemma-4-26b-a4b-ollama | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | gemma-4-26b-a4b-ollama | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | gemma-4-26b-a4b-ollama | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | gemma-4-26b-a4b-ollama | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | gemma-4-26b-a4b-ollama | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | gemma-4-26b-a4b-ollama | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | gemma-4-26b-a4b-ollama | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | gemma-4-26b-a4b-ollama | c4 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | gemma-4-26b-a4b-ollama-mlx | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | gemma-4-26b-a4b-ollama-mlx | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | gemma-4-26b-a4b-ollama-mlx | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | gemma-4-26b-a4b-ollama-mlx | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | gemma-4-26b-a4b-ollama-mlx | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | gemma-4-26b-a4b-ollama-mlx | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | gemma-4-26b-a4b-ollama-mlx | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | gemma-4-26b-a4b-ollama-mlx | c4 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | llama-3.2-3b-ollama | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | llama-3.2-3b-ollama | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | llama-3.2-3b-ollama | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | llama-3.2-3b-ollama | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | llama-3.2-3b-ollama | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | llama-3.2-3b-ollama | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | llama-3.2-3b-ollama | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | llama-3.2-3b-ollama | c4 | text-completion-bench | tp1 | not_run |  |  |
| pie | m1-max-32g | muse-glimmer-30b-mlx4 | control-aa | text-completion-bench | tp1 | not_run |  |  |
| pie | m1-max-32g | muse-glimmer-30b-mlx4 | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| pie | m1-max-32g | muse-glimmer-30b-mlx4 | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| pie | m1-max-32g | muse-glimmer-30b-mlx4 | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| pie | m1-max-32g | muse-glimmer-30b-mlx4 | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| pie | m1-max-32g | muse-glimmer-30b-mlx4 | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| pie | m1-max-32g | muse-glimmer-30b-mlx4 | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| pie | m1-max-32g | muse-glimmer-30b-mlx4 | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| pie | m1-max-32g | muse-glimmer-30b-mlx4 | c4 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | muse-glimmer-30b-ollama | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | muse-glimmer-30b-ollama | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | muse-glimmer-30b-ollama | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | muse-glimmer-30b-ollama | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | muse-glimmer-30b-ollama | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | muse-glimmer-30b-ollama | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | muse-glimmer-30b-ollama | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| ollama | m1-max-32g | muse-glimmer-30b-ollama | c4 | text-completion-bench | tp1 | not_run |  |  |
| pie | m2-max | gemma-4-26b-a4b-mlx4 | control-aa | text-completion-bench | tp1 | not_run |  |  |
| pie | m2-max | gemma-4-26b-a4b-mlx4 | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| pie | m2-max | gemma-4-26b-a4b-mlx4 | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| pie | m2-max | gemma-4-26b-a4b-mlx4 | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| pie | m2-max | gemma-4-26b-a4b-mlx4 | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| pie | m2-max | gemma-4-26b-a4b-mlx4 | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| pie | m2-max | gemma-4-26b-a4b-mlx4 | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| pie | m2-max | gemma-4-26b-a4b-mlx4 | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| pie | m2-max | gemma-4-26b-a4b-mlx4 | c4 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | gemma-4-26b-a4b-ollama | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | gemma-4-26b-a4b-ollama | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | gemma-4-26b-a4b-ollama | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | gemma-4-26b-a4b-ollama | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | gemma-4-26b-a4b-ollama | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | gemma-4-26b-a4b-ollama | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | gemma-4-26b-a4b-ollama | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | gemma-4-26b-a4b-ollama | c4 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | gemma-4-26b-a4b-ollama-mlx | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | gemma-4-26b-a4b-ollama-mlx | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | gemma-4-26b-a4b-ollama-mlx | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | gemma-4-26b-a4b-ollama-mlx | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | gemma-4-26b-a4b-ollama-mlx | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | gemma-4-26b-a4b-ollama-mlx | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | gemma-4-26b-a4b-ollama-mlx | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | gemma-4-26b-a4b-ollama-mlx | c4 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | llama-3.2-3b-ollama | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | llama-3.2-3b-ollama | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | llama-3.2-3b-ollama | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | llama-3.2-3b-ollama | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | llama-3.2-3b-ollama | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | llama-3.2-3b-ollama | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | llama-3.2-3b-ollama | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | llama-3.2-3b-ollama | c4 | text-completion-bench | tp1 | not_run |  |  |
| pie | m2-max | muse-glimmer-30b-mlx4 | control-aa | text-completion-bench | tp1 | not_run |  |  |
| pie | m2-max | muse-glimmer-30b-mlx4 | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| pie | m2-max | muse-glimmer-30b-mlx4 | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| pie | m2-max | muse-glimmer-30b-mlx4 | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| pie | m2-max | muse-glimmer-30b-mlx4 | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| pie | m2-max | muse-glimmer-30b-mlx4 | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| pie | m2-max | muse-glimmer-30b-mlx4 | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| pie | m2-max | muse-glimmer-30b-mlx4 | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| pie | m2-max | muse-glimmer-30b-mlx4 | c4 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | muse-glimmer-30b-ollama | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | muse-glimmer-30b-ollama | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | muse-glimmer-30b-ollama | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | muse-glimmer-30b-ollama | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | muse-glimmer-30b-ollama | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | muse-glimmer-30b-ollama | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | muse-glimmer-30b-ollama | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| ollama | m2-max | muse-glimmer-30b-ollama | c4 | text-completion-bench | tp1 | not_run |  |  |
| pie | m4-pro-48g | gemma-4-26b-a4b-mlx4 | control-aa | text-completion-bench | tp1 | not_run |  |  |
| pie | m4-pro-48g | gemma-4-26b-a4b-mlx4 | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| pie | m4-pro-48g | gemma-4-26b-a4b-mlx4 | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| pie | m4-pro-48g | gemma-4-26b-a4b-mlx4 | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| pie | m4-pro-48g | gemma-4-26b-a4b-mlx4 | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| pie | m4-pro-48g | gemma-4-26b-a4b-mlx4 | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| pie | m4-pro-48g | gemma-4-26b-a4b-mlx4 | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| pie | m4-pro-48g | gemma-4-26b-a4b-mlx4 | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| pie | m4-pro-48g | gemma-4-26b-a4b-mlx4 | c4 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | gemma-4-26b-a4b-ollama | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | gemma-4-26b-a4b-ollama | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | gemma-4-26b-a4b-ollama | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | gemma-4-26b-a4b-ollama | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | gemma-4-26b-a4b-ollama | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | gemma-4-26b-a4b-ollama | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | gemma-4-26b-a4b-ollama | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | gemma-4-26b-a4b-ollama | c4 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | gemma-4-26b-a4b-ollama-mlx | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | gemma-4-26b-a4b-ollama-mlx | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | gemma-4-26b-a4b-ollama-mlx | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | gemma-4-26b-a4b-ollama-mlx | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | gemma-4-26b-a4b-ollama-mlx | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | gemma-4-26b-a4b-ollama-mlx | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | gemma-4-26b-a4b-ollama-mlx | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | gemma-4-26b-a4b-ollama-mlx | c4 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | llama-3.2-3b-ollama | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | llama-3.2-3b-ollama | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | llama-3.2-3b-ollama | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | llama-3.2-3b-ollama | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | llama-3.2-3b-ollama | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | llama-3.2-3b-ollama | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | llama-3.2-3b-ollama | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | llama-3.2-3b-ollama | c4 | text-completion-bench | tp1 | not_run |  |  |
| pie | m4-pro-48g | muse-glimmer-30b-mlx4 | control-aa | text-completion-bench | tp1 | not_run |  |  |
| pie | m4-pro-48g | muse-glimmer-30b-mlx4 | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| pie | m4-pro-48g | muse-glimmer-30b-mlx4 | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| pie | m4-pro-48g | muse-glimmer-30b-mlx4 | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| pie | m4-pro-48g | muse-glimmer-30b-mlx4 | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| pie | m4-pro-48g | muse-glimmer-30b-mlx4 | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| pie | m4-pro-48g | muse-glimmer-30b-mlx4 | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| pie | m4-pro-48g | muse-glimmer-30b-mlx4 | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| pie | m4-pro-48g | muse-glimmer-30b-mlx4 | c4 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | muse-glimmer-30b-ollama | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | muse-glimmer-30b-ollama | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | muse-glimmer-30b-ollama | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | muse-glimmer-30b-ollama | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | muse-glimmer-30b-ollama | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | muse-glimmer-30b-ollama | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | muse-glimmer-30b-ollama | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| ollama | m4-pro-48g | muse-glimmer-30b-ollama | c4 | text-completion-bench | tp1 | not_run |  |  |
| pie | m5-max-128g | gemma-4-26b-a4b-mlx4 | control-aa | text-completion-bench | tp1 | not_run |  |  |
| pie | m5-max-128g | gemma-4-26b-a4b-mlx4 | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| pie | m5-max-128g | gemma-4-26b-a4b-mlx4 | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| pie | m5-max-128g | gemma-4-26b-a4b-mlx4 | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| pie | m5-max-128g | gemma-4-26b-a4b-mlx4 | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| pie | m5-max-128g | gemma-4-26b-a4b-mlx4 | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| pie | m5-max-128g | gemma-4-26b-a4b-mlx4 | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| pie | m5-max-128g | gemma-4-26b-a4b-mlx4 | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| pie | m5-max-128g | gemma-4-26b-a4b-mlx4 | c4 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | gemma-4-26b-a4b-ollama | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | gemma-4-26b-a4b-ollama | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | gemma-4-26b-a4b-ollama | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | gemma-4-26b-a4b-ollama | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | gemma-4-26b-a4b-ollama | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | gemma-4-26b-a4b-ollama | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | gemma-4-26b-a4b-ollama | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | gemma-4-26b-a4b-ollama | c4 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | gemma-4-26b-a4b-ollama-mlx | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | gemma-4-26b-a4b-ollama-mlx | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | gemma-4-26b-a4b-ollama-mlx | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | gemma-4-26b-a4b-ollama-mlx | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | gemma-4-26b-a4b-ollama-mlx | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | gemma-4-26b-a4b-ollama-mlx | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | gemma-4-26b-a4b-ollama-mlx | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | gemma-4-26b-a4b-ollama-mlx | c4 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | llama-3.2-3b-ollama | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | llama-3.2-3b-ollama | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | llama-3.2-3b-ollama | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | llama-3.2-3b-ollama | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | llama-3.2-3b-ollama | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | llama-3.2-3b-ollama | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | llama-3.2-3b-ollama | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | llama-3.2-3b-ollama | c4 | text-completion-bench | tp1 | not_run |  |  |
| pie | m5-max-128g | muse-glimmer-30b-mlx4 | control-aa | text-completion-bench | tp1 | not_run |  |  |
| pie | m5-max-128g | muse-glimmer-30b-mlx4 | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| pie | m5-max-128g | muse-glimmer-30b-mlx4 | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| pie | m5-max-128g | muse-glimmer-30b-mlx4 | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| pie | m5-max-128g | muse-glimmer-30b-mlx4 | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| pie | m5-max-128g | muse-glimmer-30b-mlx4 | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| pie | m5-max-128g | muse-glimmer-30b-mlx4 | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| pie | m5-max-128g | muse-glimmer-30b-mlx4 | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| pie | m5-max-128g | muse-glimmer-30b-mlx4 | c4 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | muse-glimmer-30b-ollama | ob-story-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | muse-glimmer-30b-ollama | ob-512-200 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | muse-glimmer-30b-ollama | ob-short-100 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | muse-glimmer-30b-ollama | ob-advanced-500 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | muse-glimmer-30b-ollama | ctx-4k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | muse-glimmer-30b-ollama | ctx-8k-256 | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | muse-glimmer-30b-ollama | cache-2k | text-completion-bench | tp1 | not_run |  |  |
| ollama | m5-max-128g | muse-glimmer-30b-ollama | c4 | text-completion-bench | tp1 | not_run |  |  |

## Declared unsupported

| reason | cells |
|---|---|
| pie ships no Llama reader | 45 |

## pie pass rate by family × platform

| family | m1-max-32g | m2-max | m4-pro-48g | m5-max-128g | m5-max-48g |
|---|---|---|---|---|---|
| gemma4_moe | 0/9 | 0/9 | 0/9 | 0/9 | 9/9 |
| muse_glimmer | 0/9 | 0/9 | 0/9 | 0/9 | 9/9 |
