"""Lazy waveform/features dataset and Whisper label padding for GPU training."""
from __future__ import annotations

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


def read_waveform(audio_root, relative_audio):
    waveform, sr = sf.read(str(audio_root / relative_audio), dtype='float32')
    if sr != 16000 or waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError(f'Waveform không hợp lệ: {relative_audio}')
    return waveform


class WhisperDataset(Dataset):
    def __init__(self, rows, processor, audio_root, max_label_length):
        self.processor = processor
        self.audio_root = audio_root
        self.max_label_length = max_label_length
        self.rows = rows
        self.labels = []
        for row in rows:
            labels = self.processor.tokenizer(row['audio_text']).input_ids
            if len(labels) > self.max_label_length:
                raise ValueError(f"{row['clip_id']}: label quá dài: {len(labels)} > {self.max_label_length}")
            self.labels.append(labels)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        features = self.processor.feature_extractor(
            read_waveform(self.audio_root, self.rows[index]['audio']), sampling_rate=16000,
            return_attention_mask=True,
        )
        return {'input_features': features.input_features[0],
                'attention_mask': features.attention_mask[0], 'labels': self.labels[index]}


class WhisperCollator:
    def __init__(self, processor, decoder_start_token_id):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features):
        batch = {
            'input_features': torch.tensor(np.stack([f['input_features'] for f in features]), dtype=torch.float32),
            'attention_mask': torch.tensor(np.stack([f['attention_mask'] for f in features]), dtype=torch.long),
        }
        padded = self.processor.tokenizer.pad([{'input_ids': f['labels']} for f in features], return_tensors='pt')
        labels = padded['input_ids'].masked_fill(padded['attention_mask'].ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
        batch['labels'] = labels
        return batch
