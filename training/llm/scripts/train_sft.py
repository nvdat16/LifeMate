"""
Unsloth Fine-Tuning Script for LifeMate Agentic Voice Assistant (Zalo AI Challenge 2026)

Supported models: Qwen2.5-3B-Instruct and Qwen3-4B-Instruct-2507 (select with --model_name)
Dataset: training/llm/data/sft/harmonized_train_chatml.jsonl (1,450 train) & harmonized_val_chatml.jsonl (76 val)
Method: QLoRA (4-bit) / LoRA with response-only loss masking (train_on_responses_only)

Features:
- Checkpoints saved approximately every 0.5 epoch; supports one or more GPUs via torchrun
- Response-only loss masking: masks system prompt, user prompt, and sandbox observations
- Full Hugging Face Token (hf_token) & Weights & Biases (wandb) support
- Automatic merge to 16-bit and optional GGUF export for fast vLLM/Ollama inference
"""

import os
import random
import re
import sys
import argparse
import math
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[2]

# Kaggle mounts project inputs read-only. Unsloth uses a relative tokenizer cache,
# so run from the writable working directory and send model/compiler caches there.
KAGGLE_WORKING_DIR = Path("/kaggle/working")
if KAGGLE_WORKING_DIR.is_dir():
    HF_CACHE_DIR = KAGGLE_WORKING_DIR / "huggingface_cache"
    HF_HUB_CACHE_DIR = HF_CACHE_DIR / "hub"
    UNSLOTH_COMPILE_CACHE_DIR = KAGGLE_WORKING_DIR / "unsloth_compiled_cache"
    HF_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    UNSLOTH_COMPILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE_DIR)
    os.environ["UNSLOTH_COMPILE_LOCATION"] = str(UNSLOTH_COMPILE_CACHE_DIR)
    os.chdir(KAGGLE_WORKING_DIR)

sys.path.insert(0, str(TRAINING_ROOT))
from paths import (PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR, LLM_OUTPUT_DIR)

WORKSPACE_ROOT = PROJECT_ROOT


def distributed_world_size() -> int:
    """Number of processes started by torchrun; one for ordinary python runs."""
    return max(1, int(os.environ.get("WORLD_SIZE", "1")))


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch before model/data initialization."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError as exc:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def resolve_chat_template(model_name: str, requested_template: str) -> str:
    if requested_template != "auto":
        return requested_template
    model_id = model_name.lower()
    if "qwen3" in model_id:
        return "qwen3-instruct"
    if "qwen2.5" in model_id or "qwen2_5" in model_id:
        return "qwen-2.5"
    raise ValueError(
        "Could not infer a chat template from --model_name. "
        "Set --chat_template explicitly (qwen-2.5 or qwen3-instruct)."
    )


def default_output_dir(model_name: str) -> Path:
    model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name.rstrip("/").split("/")[-1])
    output_root = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else LLM_OUTPUT_DIR
    return output_root / f"{model_slug.lower()}_lifemate_lora"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a supported Qwen model for LifeMate with Unsloth")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-3B-Instruct", help="Base model identifier on Hugging Face")
    parser.add_argument("--chat_template", choices=["auto", "qwen-2.5", "qwen3-instruct"], default="auto", help="Chat template; auto selects the matching template for either supported Qwen family")
    parser.add_argument("--train_file", type=str, default=str(SFT_DIR / "harmonized_train_chatml.jsonl"), help="Path to train JSONL file")
    parser.add_argument("--val_file", type=str, default=str(SFT_DIR / "harmonized_val_chatml.jsonl"), help="Path to val JSONL file")
    parser.add_argument("--max_seq_length", type=int, default=12288, help="Training context length; must not exceed the competition's 16K limit")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device train batch size (conservative for Kaggle GPUs)")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Total training epochs")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank r")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (defaults to /kaggle/working on Kaggle, otherwise training/llm/outputs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training")

    # Hugging Face & WANDB integration
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", None), help="Hugging Face API token (for gated/private models or pushing to hub)")
    parser.add_argument("--push_to_hub", action="store_true", help="Upload the final LoRA adapter to Hugging Face after training")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Hugging Face repo id, e.g. username/lifemate-qwen3-lora (required with --push_to_hub)")
    parser.add_argument("--hub_public", action="store_true", help="Create the Hugging Face repo as public; private is the default")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="lifemate-zaic-2026", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name (defaults to the selected model name)")

    args = parser.parse_args()

    if args.max_seq_length <= 0 or args.max_seq_length > 16384:
        raise ValueError("--max_seq_length must be between 1 and 16384 tokens for this competition.")
    if args.push_to_hub and not args.hub_model_id:
        parser.error("--hub_model_id is required when --push_to_hub is enabled.")
    if args.push_to_hub and not args.hf_token:
        parser.error("Set HF_TOKEN or pass --hf_token with a Hugging Face token that has write access.")

    chat_template = resolve_chat_template(args.model_name, args.chat_template)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.model_name)
    run_name = args.wandb_run_name or output_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    print("=" * 80)
    print("🚀 STARTING UNSLOTH SFT TRAINING PIPELINE")
    print(f"Base Model:       {args.model_name}")
    print(f"Chat Template:    {chat_template}")
    print(f"Train File:       {args.train_file}")
    print(f"Val File:         {args.val_file}")
    print(f"Max Seq Length:   {args.max_seq_length}")
    print(f"LoRA Rank (r):    {args.lora_rank} (Alpha: {args.lora_alpha})")
    print(f"Random Seed:      {args.seed}")
    print(f"DDP World Size:   {distributed_world_size()}")
    print(f"Effective Batch:  {args.batch_size * args.grad_accum * distributed_world_size()}")
    print(f"Output Directory: {output_dir}")
    print(f"HF Token Set:     {'Yes (Hidden)' if args.hf_token else 'No (Public Access)'}")
    print(f"WandB Logging:    {'Enabled' if args.use_wandb else 'Disabled (report_to=none)'}")
    if args.use_wandb and is_main_process():
        print(f"  - Project:      {args.wandb_project}")
        print(f"  - Run Name:     {run_name}")
    print("=" * 80)

    # Initialize WandB once; other DDP workers report through Trainer callbacks.
    if args.use_wandb and is_main_process():
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "model_name": args.model_name,
                    "chat_template": chat_template,
                    "max_seq_length": args.max_seq_length,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "world_size": distributed_world_size(),
                    "effective_batch": args.batch_size * args.grad_accum * distributed_world_size(),
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "seed": args.seed,
                    "train_file": args.train_file,
                    "val_file": args.val_file,
                }
            )
            print("✅ Weights & Biases initialized successfully!")
        except ImportError:
            print("⚠️ Warning: wandb package is not installed. Run 'pip install wandb' to log experiments.")
            args.use_wandb = False

    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template, train_on_responses_only
        from trl import SFTConfig, SFTTrainer
        from datasets import load_dataset
    except ImportError as exc:
        print("\n❌ Error: unsloth or trl is not installed in the current environment!")
        print(f"Missing training dependency: {exc}")
        print("In a Kaggle notebook, install the current stack with:")
        print("  %pip install -q --upgrade unsloth datasets trl")
        sys.exit(1)

    # 1. Load Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None, # Auto-detect (Float16 or Bfloat16)
        load_in_4bit=True, # 4-bit QLoRA to fit comfortably in 12GB - 24GB VRAM
        token=args.hf_token, # HF authentication token if needed
    )

    # 2. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0, # Unsloth supports 0 dropout optimized
        bias="none",
        use_gradient_checkpointing="unsloth", # 30% less VRAM usage
        random_state=args.seed,
    )

    # 3. Setup Chat Template
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    # 4. Load Dataset
    train_path = str(Path(args.train_file).resolve())
    val_path = str(Path(args.val_file).resolve())

    if not Path(train_path).exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not Path(val_path).exists():
        raise FileNotFoundError(f"Val file not found: {val_path}")

    dataset = load_dataset("json", data_files={"train": train_path, "val": val_path})

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 5. Training Arguments (saving/evaluating approximately every 0.5 epoch)
    steps_per_epoch = math.ceil(
        len(dataset["train"])
        / (args.batch_size * args.grad_accum * distributed_world_size())
    )
    save_steps = max(1, steps_per_epoch // 2)
    import torch
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not bf16_supported,
        bf16=bf16_supported,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=args.seed,
        data_seed=args.seed,
        output_dir=str(output_dir),
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=6, # Keep all 6 checkpoints (0.5, 1.0, 1.5, 2.0, 2.5, 3.0 epochs)
        eval_strategy="steps",
        eval_steps=save_steps,
        ddp_find_unused_parameters=False,
        report_to="wandb" if args.use_wandb else "none",
        dataset_text_field="text",
        dataset_num_proc=2,
        max_length=args.max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        args=training_args,
    )

    # 6. Response-Only Loss Masking (Crucial: only train on assistant JSON output)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # 7. Start Training
    if is_main_process():
        print("Commencing fine-tuning...")
    trainer_stats = trainer.train()
    trainer.accelerator.wait_for_everyone()

    # Only rank 0 writes shared output files or uploads to Hugging Face.
    if trainer.is_world_process_zero():
        print("Training finished successfully!")
        print(f"Stats: {trainer_stats}")

        # 8. Save final adapter.
        final_adapter_dir = output_dir / "final_adapter"
        model.save_pretrained(str(final_adapter_dir))
        tokenizer.save_pretrained(str(final_adapter_dir))
        print(f"Saved LoRA adapter to: {final_adapter_dir}")

        # 9. Optionally upload only the adapter and tokenizer.
        if args.push_to_hub:
            from huggingface_hub import HfApi

            hub_api = HfApi(token=args.hf_token)
            hub_api.create_repo(
                repo_id=args.hub_model_id,
                repo_type="model",
                private=not args.hub_public,
                exist_ok=True,
            )
            hub_api.upload_folder(
                folder_path=str(final_adapter_dir),
                repo_id=args.hub_model_id,
                repo_type="model",
                commit_message="Upload trained LifeMate LoRA adapter",
            )
            visibility = "public" if args.hub_public else "private"
            print(f"Uploaded LoRA adapter to https://huggingface.co/{args.hub_model_id} ({visibility})")

        # 10. Export merged weights once, from rank 0 only.
        merged_dir = output_dir / "merged_16bit"
        print(f"Merging LoRA weights to 16-bit at: {merged_dir}")
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        print("Model merged successfully! Ready for vLLM / Ollama.")

    trainer.accelerator.wait_for_everyone()
if __name__ == "__main__":
    main()
