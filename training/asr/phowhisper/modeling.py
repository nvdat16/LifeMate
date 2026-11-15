"""Resolve the base revision, configure Whisper decoding and attach LoRA."""
from __future__ import annotations

import json
from dataclasses import dataclass

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoProcessor, AutoModelForSpeechSeq2Seq, BitsAndBytesConfig, set_seed


@dataclass(frozen=True)
class ModelContext:
    processor: object
    model_config: object
    resolved_revision: str
    compute_dtype: torch.dtype
    previous_config: dict | None


def initialize_model(args, run_dir):
    if not torch.cuda.is_available():
        raise RuntimeError('Fine-tuning cần GPU NVIDIA/CUDA. Dùng --prepare-only để kiểm tra dữ liệu trên CPU.')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Cần đúng một GPU khả dụng; đặt CUDA_VISIBLE_DEVICES hoặc --gpu trước khi chạy.')
    set_seed(args.seed)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    existing_config = run_dir / 'training_config.json'
    previous_config = json.loads(existing_config.read_text()) if existing_config.exists() else None
    if previous_config and previous_config['base_model'] != args.base_model:
        raise ValueError('Base model khác run cũ. Đổi --run-name.')
    model_config = AutoConfig.from_pretrained(args.base_model, revision=previous_config['resolved_base_revision'] if previous_config else args.base_revision)
    resolved_revision = previous_config['resolved_base_revision'] if previous_config else (getattr(model_config, '_commit_hash', None) or args.base_revision)
    processor = AutoProcessor.from_pretrained(args.base_model, revision=resolved_revision, language='vi', task='transcribe')
    processor.tokenizer.set_prefix_tokens(language='vi', task='transcribe')
    print('GPU:', torch.cuda.get_device_name(0))
    print('Compute dtype:', compute_dtype, '| Quantization:', 'NF4 4-bit' if args.use_4bit else 'none')
    print('Resolved base commit:', resolved_revision)
    print('torch:', torch.__version__, '| transformers:', transformers.__version__, '| peft:', peft.__version__)
    return ModelContext(processor, model_config, resolved_revision, compute_dtype, previous_config)


def load_base_model(args, context):
    kwargs = {'revision': context.resolved_revision, 'torch_dtype': context.compute_dtype,
              'device_map': {'': 0}, 'attn_implementation': 'sdpa'}
    if args.use_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=context.compute_dtype,
        )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.base_model, **kwargs)
    model.generation_config.language = 'vi'
    model.generation_config.task = 'transcribe'
    model.generation_config.forced_decoder_ids = None
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1
    model.generation_config.return_dict_in_generate = False
    model.generation_config.max_new_tokens = args.max_new_tokens
    return model


def enable_conv_gradient(module, inputs, output):
    output.requires_grad_(True)


def build_lora_model(args, context):
    base_model = load_base_model(args, context)
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
    )
    base_model.config.use_cache = False

    base_model.model.encoder.conv1.register_forward_hook(enable_conv_gradient)
    model = get_peft_model(base_model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=['q_proj', 'v_proj'], bias='none',
    ))
    model.print_trainable_parameters()
    assert any(p.requires_grad for p in model.parameters()), 'Không có tham số trainable.'
    assert all('lora_' in name for name, parameter in model.named_parameters() if parameter.requires_grad), 'Base chưa được freeze hoàn toàn.'
    return model
