# LLM adapters

- `mock_llm.py`: mock engine for pipeline development; it does not call a hosted or local model.
- `local_llm_adapter.py`: OpenAI-compatible Chat Completions client. It can call Ollama/vLLM locally or a hosted API. `main.py` selects the backend and passes the endpoint/model/key.

## CLI providers

Local Ollama (default endpoint `http://localhost:11434/v1`):

```bash
python main.py --mode ollama --model qwen2.5:1.5b --split public_test --output submission.jsonl
```

OpenAI API (default endpoint `https://api.openai.com/v1`):

```bash
python main.py --mode openai --model "REPLACE_WITH_MODEL_ID" --split public_test --output submission.jsonl
```

For OpenAI mode, set `OPENAI_API_KEY` in the process environment or Kaggle Secrets. The key is not accepted as a CLI option and must not be stored in source files. Use `--base-url` for another OpenAI-compatible endpoint.

The adapter raises an error on HTTP, network, or malformed-response failures. Inference stops instead of writing a submission as if the failed LLM call had completed normally.
