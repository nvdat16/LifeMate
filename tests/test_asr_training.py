"""CPU checks for the standalone ASR trainer; no model downloads required."""
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

SCRIPT = Path(__file__).resolve().parents[1] / 'training/asr/scripts/main.py'
sys.path.insert(0, str(SCRIPT.parents[1]))

from phowhisper import artifacts, config, data, metrics, splitting



class ASRTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'train'
        self.data.mkdir()
        self.rows = []
        for i in range(12):
            audio = f'{i}.wav'
            with wave.open(str(self.data / audio), 'wb') as wav:
                wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                wav.writeframes(struct.pack('<h', i + 1) * 160)
            self.rows.append({'id': f'train_{i}', 'audio': audio,
                              'audio_text': f'Bật đèn số {i}', 'history': []})
        self.source = self.data / 'train.jsonl'
        self.write_source()

    def write_source(self):
        self.source.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in self.rows), encoding='utf-8')

    def test_cli_prepares_and_reruns_without_training_dependencies(self):
        command = [sys.executable, '-S', str(SCRIPT), '--data-root', str(self.root),
                   '--output-root', str(self.root / 'outputs'), '--prepare-only']
        for _ in range(2):
            result = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        run = self.root / 'outputs/phowhisper-large-lora-v1'
        manifest = json.loads((run / 'split_manifest.json').read_text())
        self.assertEqual(sum(len(rows) for rows in manifest['splits'].values()), 12)
        self.assertTrue(all(manifest['splits'].values()))
        self.assertTrue(splitting.audit_splits(manifest['splits']))
        self.assertFalse((run / 'training_config.json').exists())
        self.assertEqual(data.find_train(self.root), self.source.resolve())

    def test_conversation_speaker_and_duplicate_groups_remain_together(self):
        self.rows[0]['speaker_id'] = 'speaker-a'
        self.rows[1]['speaker_id'] = 'speaker-a'
        self.rows[2]['history'] = [{'role': 'user', 'audio': self.rows[0]['audio'],
                                    'audio_text': self.rows[0]['audio_text']}]
        self.write_source()
        clips = data.read_clips(self.source, self.data)
        groups, duplicates, group_ids = splitting.group_and_deduplicate(clips)
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(group_ids['train_0:current'], group_ids['train_2:current'])
        self.assertEqual(group_ids['train_0:current'], group_ids['train_1:current'])
        self.assertEqual(sum(map(len, groups.values())), 12)
        ratios = dict(train=.8, validation=.1, test=.1)
        splits = splitting.balanced_group_split(groups, ratios)
        self.assertEqual(splits, splitting.balanced_group_split(groups, ratios))
        self.assertTrue(splitting.audit_splits(splits))

    def test_duplicate_pcm_with_conflicting_transcript_is_rejected(self):
        self.rows[1]['audio'] = self.rows[0]['audio']
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'transcript khác nhau'):
            splitting.group_and_deduplicate(data.read_clips(self.source, self.data))

    def test_audio_outside_root_and_wrong_sample_rate_are_rejected(self):
        self.rows[0]['audio'] = '../outside.wav'
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'ngoài audio root'):
            data.read_clips(self.source, self.data)
        self.rows[0]['audio'] = '0.wav'
        self.write_source()
        with wave.open(str(self.data / '0.wav'), 'wb') as wav:
            wav.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
            wav.writeframes(b'\x01\x00' * 80)
        with self.assertRaisesRegex(ValueError, '16 kHz'):
            data.read_clips(self.source, self.data)

    def test_existing_run_rejects_changed_split(self):
        args = config.parse_args(['--data-root', str(self.source), '--output-root', str(self.root / 'outputs')])
        with contextlib.redirect_stdout(io.StringIO()):
            data.prepare_data(args)
            args.seed += 1
            with self.assertRaisesRegex(ValueError, 'split khác'):
                data.prepare_data(args)

    def test_restore_relocates_best_checkpoint_without_modifying_source(self):
        source = self.root / 'old'
        checkpoint = source / 'checkpoints/checkpoint-3'
        checkpoint.mkdir(parents=True)
        for name in ('split_manifest.json', 'training_config.json'):
            (source / name).write_text('{}')
        state = {'best_model_checkpoint': 'D:\\old\\checkpoints\\checkpoint-3', 'global_step': 3}
        (checkpoint / 'trainer_state.json').write_text(json.dumps(state))
        destination = self.root / 'new'
        artifacts.restore_run(source, destination)
        restored = json.loads((destination / 'checkpoints/checkpoint-3/trainer_state.json').read_text())
        self.assertEqual(restored['best_model_checkpoint'], str((destination / 'checkpoints/checkpoint-3').resolve()))
        self.assertEqual(json.loads((checkpoint / 'trainer_state.json').read_text()), state)
        with self.assertRaisesRegex(ValueError, 'đích mới'):
            artifacts.restore_run(source, destination)

    def test_metrics_use_corpus_totals_and_preserve_vietnamese(self):
        self.assertEqual(metrics.metric_text('  BẬT, đèn!  '), 'bật đèn')
        self.assertEqual(metrics.score_pair('Bật đèn!', 'bật đèn')['wer'], 0)
        self.assertGreater(metrics.score_pair('đèn', 'den')['cer'], 0)
        result = metrics.corpus_metrics(['một', 'hai ba bốn'], ['sai', 'hai ba bốn'])
        self.assertEqual(result['wer'], .25)
        self.assertEqual(result['exact_match'], .5)

    def test_env_default_and_invalid_arguments(self):
        with patch.dict(os.environ, {'ZAIC_DATA_ROOT': str(self.root)}):
            self.assertEqual(config.parse_args([]).data_root, str(self.root))
        for arguments in (['--batch-size', '0'], ['--epochs', 'nan'],
                          ['--run-name', '../escape'], ['--max-new-tokens', '441'],
                          ['--split-ratios', '.8', '.2', '.1']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                config.parse_args(arguments)


if __name__ == '__main__':
    unittest.main()
