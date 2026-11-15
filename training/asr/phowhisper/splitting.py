"""Group conversations, speakers and exact PCM duplicates before splitting."""
from __future__ import annotations

import math
import random
from collections import defaultdict

from .metrics import metric_text


class UnionFind:
    def __init__(self, count):
        self.parent = list(range(count))
    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index
    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def group_and_deduplicate(clips, deduplicate=True):
    uf = UnionFind(len(clips))
    owners = {}
    hashes = defaultdict(list)
    for i, clip in enumerate(clips):
        keys = [('sample', clip['id']), ('pcm', clip['pcm_sha256'])]
        if clip['speaker_id'] is not None:
            keys.append(('speaker', clip['speaker_id']))
        for key in keys:
            if key in owners:
                uf.union(i, owners[key])
            else:
                owners[key] = i
        hashes[clip['pcm_sha256']].append(i)
    duplicates = []
    dropped = set()
    for pcm_hash, indices in hashes.items():
        if len(indices) < 2:
            continue
        if len({metric_text(clips[i]['audio_text']) for i in indices}) != 1:
            raise ValueError(f'Audio PCM trùng nhưng transcript khác nhau: {[clips[i]["clip_id"] for i in indices]}')
        ordered = sorted(indices, key=lambda i: (clips[i]['kind'] != 'current', clips[i]['clip_id']))
        duplicates.append({'pcm_sha256': pcm_hash, 'kept': clips[ordered[0]]['clip_id'],
                           'duplicates': [clips[i]['clip_id'] for i in ordered[1:]]})
        if deduplicate:
            dropped.update(ordered[1:])
    groups = defaultdict(list)
    all_group_ids = {}
    for i, clip in enumerate(clips):
        group_id = f'group-{uf.find(i):06d}'
        all_group_ids[clip['clip_id']] = group_id
        if i not in dropped:
            groups[group_id].append({**clip, 'group_id': group_id})
    return dict(groups), duplicates, all_group_ids


def balanced_group_split(groups, ratios, seed=42, restarts=24):
    names = list(ratios)
    if set(names) != {'train', 'validation', 'test'} or not math.isclose(sum(ratios.values()), 1.0) or any(v <= 0 for v in ratios.values()):
        raise ValueError('Cần ba tỷ lệ train/validation/test > 0, tổng bằng 1.')
    if len(groups) < 3:
        raise ValueError('Cần ít nhất 3 nhóm độc lập để chia train/validation/test mà không tách speaker/PCM/hội thoại.')
    vectors = {gid: [len(rows), sum(r['kind']=='current' for r in rows), sum(r['kind']=='history' for r in rows), sum(r['audio_seconds'] for r in rows)] for gid, rows in groups.items()}
    totals = [sum(v[d] for v in vectors.values()) for d in range(4)]
    scales = [max(total, 1) for total in totals]
    target = {name: [ratios[name]*v for v in totals] for name in names}
    best = None
    for attempt in range(max(1, restarts)):
        rng = random.Random(seed + attempt)
        jitter = {gid: rng.uniform(.8, 1.2) for gid in groups}
        order = sorted(groups, key=lambda gid: (-vectors[gid][0]*jitter[gid], gid))
        counts = {name: [0.0]*4 for name in names}
        assignment = {}
        for gid in order:
            def delta(name):
                return sum(((counts[name][d]+vectors[gid][d]-target[name][d])/scales[d])**2 - ((counts[name][d]-target[name][d])/scales[d])**2 for d in range(4))
            name = min(names, key=lambda name: (delta(name), names.index(name)))
            assignment[gid] = name
            counts[name] = [counts[name][d]+vectors[gid][d] for d in range(4)]
        # Greedy largest-first can overfill train with history-rich dialogues.
        # Improve by moving WHOLE groups, never breaking speaker/PCM boundaries.
        for _ in range(20):
            changed = False
            move_order = list(groups)
            rng.shuffle(move_order)
            for gid in move_order:
                origin = assignment[gid]
                vec = vectors[gid]
                if counts[origin][0] <= vec[0]:
                    continue
                def move_delta(destination):
                    return sum(
                        ((counts[origin][d]-vec[d]-target[origin][d])/scales[d])**2
                        - ((counts[origin][d]-target[origin][d])/scales[d])**2
                        + ((counts[destination][d]+vec[d]-target[destination][d])/scales[d])**2
                        - ((counts[destination][d]-target[destination][d])/scales[d])**2
                        for d in range(4)
                    )
                candidates = [(move_delta(destination), destination) for destination in names if destination != origin]
                improvement, destination = min(candidates)
                if improvement < -1e-12:
                    assignment[gid] = destination
                    counts[origin] = [counts[origin][d]-vec[d] for d in range(4)]
                    counts[destination] = [counts[destination][d]+vec[d] for d in range(4)]
                    changed = True
            if not changed:
                break
        if any(counts[name][0] == 0 for name in names):
            continue
        cost = sum(((counts[name][d]-target[name][d])/scales[d])**2 for name in names for d in range(4))
        if best is None or cost < best[0]:
            best = cost, assignment
    if best is None:
        raise ValueError('Không tạo được ba split khác rỗng; kiểm tra số speaker/nhóm độc lập.')
    splits = {name: [] for name in names}
    for gid, name in best[1].items():
        splits[name].extend(groups[gid])
    for name in names:
        splits[name].sort(key=lambda row: row['clip_id'])
    return splits


def audit_splits(splits):
    for field in ('clip_id', 'id', 'group_id', 'pcm_sha256', 'speaker_id'):
        ownership = {}
        for split, rows in splits.items():
            for row in rows:
                value = row.get(field)
                if value is None:
                    continue
                if value in ownership and ownership[value] != split:
                    raise ValueError(f'Leakage {field}: {value} xuất hiện ở {ownership[value]} và {split}.')
                ownership[value] = split
    return True


def split_statistics(splits):
    return [{
        'split': name, 'clips': len(rows), 'current': sum(r['kind']=='current' for r in rows),
        'history': sum(r['kind']=='history' for r in rows),
        'current_ratio': sum(r['kind']=='current' for r in rows)/len(rows),
        'hours': sum(r['audio_seconds'] for r in rows)/3600,
        'groups': len({r['group_id'] for r in rows}),
        'known_speakers': len({r['speaker_id'] for r in rows if r['speaker_id'] is not None}),
        'unknown_speaker_clips': sum(r['speaker_id'] is None for r in rows),
    } for name, rows in splits.items()]
