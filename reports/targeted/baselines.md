# pie vs baselines (latest, best-of-recipe)

| platform | artifact | workload | program | mode | baseline | metric | pie | baseline | baseline/pie | inputs match | recipe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| m5-max-48g | gemma-4-26b-a4b-mlx4 | cache-2k | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama 0.34.4 | output_tok_s | 34.11 | 97.56 | 2.861× **pie trails** | **NO** | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | ctx-8k-256 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama-mlx 0.34.4 | output_tok_s | 28.43 | 48.15 | 1.694× **pie trails** | yes | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | ctx-4k-256 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama-mlx 0.34.4 | output_tok_s | 51.26 | 71.28 | 1.391× **pie trails** | **NO** | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | ctx-8k-256 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama 0.34.4 | output_tok_s | 28.43 | 38.67 | 1.360× **pie trails** | **NO** | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | ob-512-200 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama 0.34.4 | output_tok_s | 109.3 | 122.8 | 1.124× **pie trails** | yes | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | ob-512-200 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama-mlx 0.34.4 | output_tok_s | 109.3 | 115.8 | 1.060× **pie trails** | **NO** | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | ob-short-100 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama-mlx 0.34.4 | output_tok_s | 137.1 | 127.3 | 0.928× | **NO** | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | ob-short-100 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama 0.34.4 | output_tok_s | 137.1 | 123.1 | 0.897× | **NO** | competitive |
| m5-max-48g | muse-glimmer-30b-mlx4 | ob-short-100 | text-completion-bench | tp1 | muse-glimmer-30b-ollama 0.34.4 | output_tok_s | 32.64 | 28.38 | 0.869× | **NO** | competitive |
| m5-max-48g | muse-glimmer-30b-mlx4 | ob-story-200 | text-completion-bench | tp1 | muse-glimmer-30b-ollama 0.34.4 | output_tok_s | 30.81 | 25.68 | 0.834× | **NO** | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | ob-story-200 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama-mlx 0.34.4 | output_tok_s | 137 | 113.6 | 0.829× | **NO** | competitive |
| m5-max-48g | muse-glimmer-30b-mlx4 | cache-2k | text-completion-bench | tp1 | muse-glimmer-30b-ollama 0.34.4 | output_tok_s | 29.89 | 24.71 | 0.827× | yes | competitive |
| m5-max-48g | muse-glimmer-30b-mlx4 | ob-512-200 | text-completion-bench | tp1 | muse-glimmer-30b-ollama 0.34.4 | output_tok_s | 32.44 | 26.67 | 0.822× | yes | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | ob-advanced-500 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama-mlx 0.34.4 | output_tok_s | 132.2 | 107.8 | 0.815× | **NO** | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | ob-advanced-500 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama 0.34.4 | output_tok_s | 132.2 | 90.9 | 0.688× | **NO** | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | c4 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama 0.34.4 | output_tok_s | 302.1 | 172.8 | 0.572× | **NO** | competitive |
| m5-max-48g | muse-glimmer-30b-mlx4 | ctx-4k-256 | text-completion-bench | tp1 | muse-glimmer-30b-ollama 0.34.4 | output_tok_s | 28.08 | 13.71 | 0.488× | yes | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | c4 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama-mlx 0.34.4 | output_tok_s | 302.1 | 122.6 | 0.406× | **NO** | competitive |
| m5-max-48g | muse-glimmer-30b-mlx4 | ctx-8k-256 | text-completion-bench | tp1 | muse-glimmer-30b-ollama 0.34.4 | output_tok_s | 25.56 | 9.652 | 0.378× | yes | competitive |
