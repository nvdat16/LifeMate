"""
Local Scorer implementing the exact Zalo AI Challenge evaluation protocol.

Evaluates predictions against train ground truth:
- Matches tool names and arguments (exact for discrete values, Token-F1 > 0.6 for free-text).
- Matches state_hash computed via zaic_tools.utils.state_hash.
- Handles Out-of-Scope (gold_calls == []) awarding 1.0 only if pred_calls == [].
- Reports official Leaderboard Score: (100 / N) * sum(Score_i).
"""

import json
import sys
from pathlib import Path
from typing import Union, Any, Dict, List, Optional
from .metrics import compute_sample_score

class LocalScorer:
    """Evaluates submission lines against train ground truth."""

    def __init__(self, data_root_or_file: Union[str, Path] = "."):
        path = Path(data_root_or_file).resolve()
        if path.is_file():
            self.train_jsonl = path
            self.data_root = path.parent.parent
        else:
            self.data_root = path
            self.train_jsonl = self.data_root / "train" / "train.jsonl"
        self._ensure_zaic_tools_imported()
        self.ground_truth = self._load_ground_truth()

    def _ensure_zaic_tools_imported(self):
        for candidate in [self.data_root / "train", self.data_root / "public_test"]:
            if (candidate / "zaic_tools").is_dir():
                c_str = str(candidate)
                if c_str not in sys.path:
                    sys.path.insert(0, c_str)

    def _load_ground_truth(self) -> dict[str, dict]:
        """Loads and pre-indexes ground truth tool calls and state hashes."""
        from zaic_tools.utils import state_hash

        mapping = {}
        if not self.train_jsonl.is_file():
            print(f"⚠️ Ground truth file not found at: {self.train_jsonl}")
            return mapping

        with open(self.train_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                s_id = d["id"]
                label = d.get("label") or {}
                final_state = label.get("final_state")
                gold_calls = label.get("tool_calls", [])

                if final_state:
                    gold_h = state_hash(final_state)
                else:
                    # If out-of-scope or unmutated, gold state is the initial environment state
                    init_state_path = self.data_root / "train" / "train_environment" / s_id / "state.json"
                    if init_state_path.is_file():
                        with open(init_state_path, "r", encoding="utf-8") as sf:
                            gold_h = state_hash(json.load(sf))
                    else:
                        gold_h = ""

                mapping[s_id] = {
                    "tool_calls": gold_calls,
                    "gold_hash": gold_h,
                    "is_out_of_scope": (len(gold_calls) == 0),
                    "audio_text": d.get("audio_text", "")
                }
        return mapping

    def evaluate(self, submission_path: Union[str, Path]) -> dict[str, Any]:
        """
        Calculates official leaderboard score and breakdown for a submission JSONL file.
        """
        sub_path = Path(submission_path)
        scores = []
        state_matches = 0
        oos_correct = 0
        total_oos = 0
        in_scope_scores = []
        precisions = []
        recalls = []

        with open(sub_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                sub_item = json.loads(line)
                s_id = sub_item["id"]

                if s_id not in self.ground_truth:
                    continue

                gt_item = self.ground_truth[s_id]
                gold_calls = gt_item["tool_calls"]
                gold_hash = gt_item["gold_hash"]

                pred_calls = sub_item.get("tool_calls", [])
                pred_hash = sub_item.get("state_hash", "")

                # Evaluate single sample
                res = compute_sample_score(
                    pred_calls=pred_calls,
                    gold_calls=gold_calls,
                    pred_hash=pred_hash,
                    gold_hash=gold_hash
                )

                score = res["score"]
                scores.append(score)
                precisions.append(res.get("precision", 0.0))
                recalls.append(res.get("recall", 0.0))

                if res["state_match"]:
                    state_matches += 1

                if gt_item["is_out_of_scope"]:
                    total_oos += 1
                    if score == 1.0:
                        oos_correct += 1
                else:
                    in_scope_scores.append(score)

        n = len(scores)
        if n == 0:
            return {"leaderboard_score": 0.0, "total_samples": 0}

        leaderboard_score = (100.0 / n) * sum(scores)
        state_match_rate = (state_matches / n) * 100.0
        avg_precision = sum(precisions) / n
        avg_recall = sum(recalls) / n
        oos_accuracy = (oos_correct / total_oos * 100.0) if total_oos > 0 else 100.0
        avg_in_scope = (sum(in_scope_scores) / len(in_scope_scores) * 100.0) if in_scope_scores else 0.0

        return {
            "leaderboard_score": round(leaderboard_score, 4),
            "state_match_rate_pct": round(state_match_rate, 2),
            "avg_precision": round(avg_precision, 4),
            "avg_recall": round(avg_recall, 4),
            "out_of_scope_accuracy_pct": round(oos_accuracy, 2),
            "in_scope_avg_score": round(avg_in_scope, 2),
            "total_evaluated": n,
            "out_of_scope_count": total_oos
        }
