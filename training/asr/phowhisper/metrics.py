"""Vietnamese text normalization, corpus WER/CER and Trainer metric callback."""
from __future__ import annotations

import unicodedata


def metric_text(text):
    text = unicodedata.normalize('NFC', text).lower()
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text).split())


def distance(left, right):
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right)+1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j]+1, current[j-1]+1, previous[j-1]+(a != b)))
        previous = current
    return previous[-1]


def score_pair(reference, prediction):
    ref, pred = metric_text(reference), metric_text(prediction)
    if not ref:
        raise ValueError('Reference rỗng sau chuẩn hóa.')
    words, chars = ref.split(), ref.replace(' ', '')
    we, ce = distance(words, pred.split()), distance(chars, pred.replace(' ', ''))
    return {'word_errors': we, 'reference_words': len(words), 'char_errors': ce, 'reference_chars': len(chars),
            'wer': we/len(words), 'cer': ce/len(chars), 'exact_match': ref == pred}


def corpus_metrics(references, predictions):
    if len(references) != len(predictions) or not references:
        raise ValueError('Reference/prediction phải cùng số lượng, khác rỗng.')
    rows = [score_pair(ref, pred) for ref, pred in zip(references, predictions)]
    return {'wer': sum(r['word_errors'] for r in rows)/sum(r['reference_words'] for r in rows),
            'cer': sum(r['char_errors'] for r in rows)/sum(r['reference_chars'] for r in rows),
            'exact_match': sum(r['exact_match'] for r in rows)/len(rows), 'clips': len(rows)}


def compute_metrics(prediction, processor):
    import numpy as np

    ids = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
    ids = np.where(ids == -100, processor.tokenizer.pad_token_id, ids)
    labels = np.where(prediction.label_ids == -100, processor.tokenizer.pad_token_id, prediction.label_ids)
    predicted = processor.batch_decode(ids, skip_special_tokens=True)
    references = processor.batch_decode(labels, skip_special_tokens=True)
    return corpus_metrics(references, predicted)
