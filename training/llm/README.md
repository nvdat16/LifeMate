# LLM training

`train_sft.py` fine-tunes the harmonized ChatML dataset with QLoRA. The base model is downloaded from Hugging Face on first use (or loaded from a local model directory).

## Supported model choices

```bash
--model_name Qwen/Qwen3-4B-Instruct-2507
--model_name unsloth/Qwen2.5-3B-Instruct
```

The trainer selects `qwen3-instruct` or `qwen-2.5` automatically. Override with `--chat_template` if using a local model path that does not include its family name. Training context is capped at 16,384 tokens; the default is 12,288. The training files default to `data/sft/harmonized_train_chatml.jsonl` and `data/sft/harmonized_val_chatml.jsonl`.

## Start a Kaggle run

Use a Kaggle Notebook with GPU enabled, Internet enabled for the initial package/model download, and this project plus its SFT files attached. If the project is attached under `/kaggle/input`, run the script from that project directory; outputs default to writable `/kaggle/working/`.

On Kaggle, `train_sft.py` automatically switches its working directory and Hugging Face/Unsloth caches to writable locations under `/kaggle/working`, while keeping the input dataset paths resolved from the script location.

Install the training dependencies in a notebook cell:

```python
%pip install -q --upgrade unsloth datasets trl
```

Start Qwen3-4B LoRA training from the project root. For a Kaggle `T4 x2` session, use `torchrun` to launch one DDP worker per GPU:

```bash
!torchrun --standalone --nproc_per_node=2 training/llm/scripts/train_sft.py \
  --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --chat_template qwen3-instruct \
  --max_seq_length 8192 \
  --batch_size 1 \
  --grad_accum 4 \
  --epochs 3 \
  --output_dir /kaggle/working/qwen3_4b_lifemate_lora
```

The two workers use data parallelism: each GPU loads its own model copy and processes different samples. With per-device batch 1 and gradient accumulation 4, the global effective batch stays 8 (`1 × 4 × 2`). Use `--grad_accum 8` for a global batch of 16. Multi-GPU can improve throughput, but it does not combine the GPUs' VRAM.

For the Qwen2.5 baseline, use `--model_name unsloth/Qwen2.5-3B-Instruct` and `--chat_template qwen-2.5`. The script saves periodic checkpoints, the final LoRA adapter, and a merged 16-bit model under the output directory. Copy or download `/kaggle/working/qwen3_4b_lifemate_lora/` before the Kaggle session ends.

The 4-bit load reduces training memory but does not change the base model's parameter count for competition eligibility. Training is not inference: download packages and weights before competition execution, then run the submitted system offline with no external API.

## Upload the adapter to Hugging Face after training

Add these options to the training command to upload `final_adapter/` when training finishes:

```bash
--push_to_hub --hub_model_id YOUR_HF_USERNAME/lifemate-qwen3-lora
```

Save a Hugging Face access token with **write** permission as a Kaggle Secret named `HF_TOKEN`, then expose it to the training process in a notebook cell:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

Add `--push_to_hub --hub_model_id YOUR_HF_USERNAME/lifemate-qwen3-lora` to the training command. The repository is private by default; add `--hub_public` only if you want to publish it. This uploads the LoRA adapter and tokenizer, not the merged 16-bit model or the base model. The adapter can then be loaded with its base model using PEFT/Transformers or Unsloth.

**Context-limit eligibility check:** the trainer and evaluator cap the configured sequence length at 16,384 tokens. The Qwen3-4B-Instruct-2507 model card lists a native context length of 262,144 tokens. A runtime cap does not change a model's native capacity; if the competition interprets its 16K rule that way, check the native context limit of either candidate and confirm eligibility with the organizers before submitting.

`evaluate_checkpoint.py` evaluates a saved checkpoint against the validation set and sandbox. See `python training/llm/scripts/train_sft.py --help` for all options.
