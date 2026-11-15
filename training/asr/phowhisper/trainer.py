"""Validate run configuration, resume training and save the best LoRA adapter."""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import peft
import torch
import transformers
from transformers import EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from .dataset import WhisperCollator, WhisperDataset
from .metrics import compute_metrics
from .modeling import build_lora_model, initialize_model


def check_encoder_gradient(model, first_batch, compute_dtype):
    # Kiểm tra một forward/backward trước khi train thật, chưa cập nhật optimizer.
    model.train()
    probe = {key: value.to('cuda:0') for key, value in first_batch.items()}
    with torch.autocast('cuda', dtype=compute_dtype):
        probe_loss = model(**probe).loss
    if not torch.isfinite(probe_loss):
        raise RuntimeError('Loss không hữu hạn ở probe batch.')
    probe_loss.backward()
    encoder_gradient = any(p.grad is not None and torch.isfinite(p.grad).all().item() and p.grad.abs().sum().item() > 0
                           for name, p in model.named_parameters() if 'encoder' in name and 'lora_' in name)
    if not encoder_gradient:
        raise RuntimeError('Encoder LoRA chưa nhận gradient; kiểm tra gradient checkpointing/hook.')
    print('Probe loss:', float(probe_loss.detach()), '| Encoder LoRA gradient:', encoder_gradient)
    model.zero_grad(set_to_none=True)
    del probe, probe_loss


def configure_training(args, data, context):
    training_config = {
        'format': 'phowhisper_lora_train_v1', 'base_model': args.base_model,
        'resolved_base_revision': context.resolved_revision, 'split_fingerprint': data.split_fingerprint,
        'smoke_test': args.smoke_test, 'use_4bit': args.use_4bit, 'compute_dtype': str(context.compute_dtype),
        'lora_r': args.lora_r, 'lora_alpha': args.lora_alpha, 'lora_dropout': args.lora_dropout,
        'target_modules': ['q_proj', 'v_proj'], 'seed': args.seed,
        'batch_size': args.batch_size, 'gradient_accumulation': args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate, 'epochs': args.epochs, 'max_new_tokens': args.max_new_tokens,
    }
    if context.previous_config is not None and context.previous_config != training_config:
        raise ValueError('Training config khác run cũ. Đổi --run-name để tránh trộn checkpoint.')
    (data.run_dir / 'training_config.json').write_text(json.dumps(training_config, indent=2)+'\n', encoding='utf-8')
    checkpoint_dir = data.run_dir / 'checkpoints'
    resume = (str(args.resume_from_checkpoint.expanduser().resolve()) if args.resume_from_checkpoint else None) or (get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None)
    if resume is not None:
        resume_path = Path(resume).resolve()
        if not resume_path.is_dir() or not resume_path.is_relative_to(checkpoint_dir.resolve()):
            raise ValueError('Checkpoint phải thuộc checkpoints/ của run đã kiểm tra fingerprint. Dùng --restore-run-dir để khôi phục run cũ trước.')
    return checkpoint_dir, resume


def train(args, data):
    context = initialize_model(args, data.run_dir)
    checkpoint_dir, resume = configure_training(args, data, context)
    train_dataset = WhisperDataset(data.work_splits['train'], context.processor, data.audio_root, context.model_config.max_target_positions)
    validation_dataset = WhisperDataset(data.work_splits['validation'], context.processor, data.audio_root, context.model_config.max_target_positions)
    collator = WhisperCollator(context.processor, context.model_config.decoder_start_token_id)
    first_batch = collator([train_dataset[0]])
    print({key: tuple(value.shape) for key, value in first_batch.items()})
    model = build_lora_model(args, context)
    check_encoder_gradient(model, first_batch, context.compute_dtype)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, warmup_ratio=0.05, weight_decay=0.01,
        num_train_epochs=args.epochs, max_steps=3 if args.smoke_test else -1,
        fp16=context.compute_dtype == torch.float16, bf16=context.compute_dtype == torch.bfloat16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model='wer', greater_is_better=False,
        predict_with_generate=True, generation_max_length=448,
        logging_steps=1 if args.smoke_test else 10,
        dataloader_num_workers=0, remove_unused_columns=False, label_names=['labels'],
        optim='adamw_torch', report_to=[], seed=args.seed, data_seed=args.seed,
        eval_accumulation_steps=1,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=validation_dataset,
        data_collator=collator, compute_metrics=partial(compute_metrics, processor=context.processor), processing_class=context.processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    train_result = trainer.train(resume_from_checkpoint=resume)
    print('Best checkpoint:', trainer.state.best_model_checkpoint)
    print('Best validation WER:', trainer.state.best_metric)
    adapter_dir = data.run_dir / 'adapter'
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    context.processor.save_pretrained(adapter_dir)
    trainer.save_state()
    train_state = {
        'best_checkpoint': trainer.state.best_model_checkpoint,
        'best_validation_wer': trainer.state.best_metric,
        'train_metrics': train_result.metrics, 'global_step': trainer.state.global_step,
        'evaluation_status': data.manifest['evaluation_status'], 'smoke_test': args.smoke_test,
        'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__,
        'gpu': torch.cuda.get_device_name(0),
    }
    (data.run_dir / 'training_result.json').write_text(json.dumps(train_state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Adapter:', adapter_dir)
    return context, adapter_dir
