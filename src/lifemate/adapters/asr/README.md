# ASR adapters

- `mock_asr.py`: instant mock for local development.
- `whisper_adapter.py`: local PhoWhisper/faster-whisper inference.
- `vieneu_asr.py`: VieNeu cloud transcription using `POST /api/v1/dub`.

## VieNeu

VieNeu's published API currently exposes transcription through its dubbing route,
not a standalone transcription endpoint. The route requires both an API key and
a valid voice ID, and it generates dubbed audio as well as returning a transcript.
VieNeu bills this request based on the generated audio duration. The adapter only
returns the `transcript` field and discards the dubbing audio URL.

In Kaggle, save the key as a Kaggle Secret named `VIENEU_API_KEY`, then expose it
to the process environment from a notebook cell:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["VIENEU_API_KEY"] = UserSecretsClient().get_secret("VIENEU_API_KEY")
```

Run inference with a voice ID available to your VieNeu account:

```bash
python main.py --split public_test --mode ollama --model qwen2.5:1.5b \
  --asr vieneu --vieneu-voice-id "<voice-id>" --output submission.jsonl
```

The documented route accepts audio files up to 10 MB. Audio larger than this is
rejected locally before sending. On `train`, existing ground-truth transcripts
are used unless `--force-asr` is supplied.
