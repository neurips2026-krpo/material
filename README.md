# NeurIPS Supplementary Material Usage

This directory has been sanitized for public release. No personal IP addresses, passwords, or machine-specific model paths are kept in the code.

## Released Model

The `qwen3_14b_krpo` model trained in this study has been publicly released on Hugging Face.

- Model page: [NIPS2026-Review-Materials/qwen3_14b_krpo](https://huggingface.co/NIPS2026-Review-Materials/qwen3_14b_krpo)
- Download: use the Hugging Face page above to access the model weights and usage instructions.

Install all dependencies with one command:

```bash
pip install -r requirements.txt
```

If you only need the evaluation scripts:

```bash
pip install -r requirements-eval.txt
```

If you only need the attention visualization script:

```bash
pip install -r requirements-attention.txt
```

The evaluation scripts use OpenAI-compatible APIs. Safe public placeholders are used by default:

- `LLM_BASE_URL=http://127.0.0.1:8000/v1`
- `LLM_API_KEY=EMPTY`
- `EMBEDDING_API_URL=http://127.0.0.1:8000/v1/embeddings`

If your local or remote service uses different endpoints, override them either with environment variables or command-line arguments.

Example:

```bash
export LLM_BASE_URL="http://127.0.0.1:8000/v1"
export LLM_API_KEY="EMPTY"
export LLM_MODEL="qwen3"
export EMBEDDING_API_URL="http://127.0.0.1:8000/v1/embeddings"
export EMBEDDING_MODEL="text-embedding-3-large"

python evaluate_model_final.py --max_samples 10
python evaluate_model_no_reasoning.py --max_samples 10
```

For the attention visualization script, the model path is intentionally not hardcoded. Provide your own path or model identifier when running it:

```bash
python plot_attention_decay.py --model_path /path/to/your/model
```

Optional arguments for `plot_attention_decay.py`:

- `--prompt_file`: load the prompt from a UTF-8 text file instead of the built-in example.
- `--output_dir`: choose where the generated figures are written.
- `--max_new_tokens`: control how many generated tokens are analyzed.
