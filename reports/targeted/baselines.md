# pie vs baselines (latest, best-of-recipe)

| platform | artifact | workload | program | mode | baseline | metric | pie | baseline | baseline/pie | inputs match | recipe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| m5-max-48g | gemma-4-26b-a4b-mlx4 | cache-2k | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama 0.34.4 | output_tok_s | 34.11 | 97.56 | 2.861× **pie trails** | **NO** | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | ctx-8k-256 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama-mlx 0.34.4 | output_tok_s | 26.66 | 48.15 | 1.806× **pie trails** | yes | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | ctx-4k-256 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama-mlx 0.34.4 | output_tok_s | 48.69 | 71.28 | 1.464× **pie trails** | **NO** | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | ctx-8k-256 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama 0.34.4 | output_tok_s | 26.66 | 38.67 | 1.451× **pie trails** | **NO** | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | ob-512-200 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama 0.34.4 | output_tok_s | 107.7 | 122.8 | 1.140× **pie trails** | yes | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | ob-512-200 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama-mlx 0.34.4 | output_tok_s | 107.7 | 115.8 | 1.075× **pie trails** | **NO** | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | ob-short-100 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama-mlx 0.34.4 | output_tok_s | 139.1 | 127.3 | 0.915× | **NO** | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | ob-short-100 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama 0.34.4 | output_tok_s | 139.1 | 123.1 | 0.885× | **NO** | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | ob-story-200 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama-mlx 0.34.4 | output_tok_s | 133.3 | 113.6 | 0.852× | **NO** | competitive |
| m5-max-48g | muse-glimmer-30b-mlx4 | cache-2k | text-completion-bench | tp1 | muse-glimmer-30b-ollama 0.34.4 | output_tok_s | 29.89 | 24.71 | 0.827× | yes | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | c4 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama 0.34.4 | output_tok_s | 304.5 | 172.8 | 0.568× | **NO** | competitive |
| m5-max-48g | gemma-4-26b-a4b-mlx4 | c4 | text-completion-bench | tp1 | gemma-4-26b-a4b-ollama-mlx 0.34.4 | output_tok_s | 304.5 | 122.6 | 0.403× | **NO** | competitive |
