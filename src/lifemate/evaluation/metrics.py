"""Official evaluation metric calculations matching Zalo AI Challenge portal."""

import re
import unicodedata
from typing import Any

# Catalog FREE_TEXT_KEYS that use fuzzy token-F1 match
FREE_TEXT_KEYS = {
    "address", "body", "content", "dropoff", "expression",
    "label", "location", "name", "note", "pickup", "subject", "title"
}

def normalize_text_nfc(s: Any) -> str:
    """Lowercases, collapses whitespace, and applies Unicode NFC."""
    if s is None:
        return ""
    return " ".join(unicodedata.normalize("NFC", str(s)).lower().split())

def fold_vietnamese(s: Any) -> str:
    """Removes Vietnamese diacritics, lowercases, collapses whitespace."""
    if s is None:
        return ""
    t = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())

def extract_tokens(s: Any) -> set[str]:
    """Extracts alphanumeric word tokens from folded text."""
    folded = fold_vietnamese(s)
    return set(re.findall(r"[^\W_]+", folded))

def compute_token_f1(pred: Any, gold: Any) -> float:
    """Computes Token-F1 between two fuzzy argument strings."""
    tokens_p = extract_tokens(pred)
    tokens_g = extract_tokens(gold)

    if not tokens_p and not tokens_g:
        return 1.0
    if not tokens_p or not tokens_g:
        return 0.0

    intersection = len(tokens_p.intersection(tokens_g))
    if intersection == 0:
        return 0.0

    precision = intersection / len(tokens_p)
    recall = intersection / len(tokens_g)
    return (2.0 * precision * recall) / (precision + recall)

def call_matches(pred_call: dict[str, Any], gold_call: dict[str, Any]) -> bool:
    """
    Checks if a predicted tool call matches a gold call according to BTC rules:
    - Same tool name.
    - Every non-fuzzy argument is identical.
    - Every fuzzy argument has token-F1 > 0.6.
    """
    if pred_call.get("name") != gold_call.get("name"):
        return False

    pred_args = pred_call.get("arguments", {})
    gold_args = gold_call.get("arguments", {})

    all_keys = set(pred_args.keys()).union(set(gold_args.keys()))

    for key in all_keys:
        val_p = pred_args.get(key)
        val_g = gold_args.get(key)

        if key in FREE_TEXT_KEYS:
            f1 = compute_token_f1(val_p, val_g)
            if f1 <= 0.6:
                return False
        else:
            # Non-fuzzy argument must be identical
            if normalize_text_nfc(val_p) != normalize_text_nfc(val_g):
                return False

    return True

def compute_sample_score(
    pred_calls: list[dict[str, Any]],
    gold_calls: list[dict[str, Any]],
    pred_hash: str,
    gold_hash: str
) -> dict[str, Any]:
    """
    Computes score for a single sample according to the official formula.
    
    Formula:
    If gold is empty:
        Score = 1 if pred is empty else 0
    Else:
        Score = (2 * P * R) / (P + R) if (pred_hash == gold_hash and M > 0) else 0
    """
    # 1. Out-of-Scope case
    if len(gold_calls) == 0:
        score = 1.0 if len(pred_calls) == 0 else 0.0
        return {
            "score": score,
            "state_match": (pred_hash == gold_hash),
            "matched_calls": 0,
            "gold_calls": 0,
            "unmatched_calls": len(pred_calls),
            "precision": 1.0 if score == 1.0 else 0.0,
            "recall": 1.0 if score == 1.0 else 0.0,
            "f1": score
        }

    # 2. Match tool calls greedily (order is not considered)
    matched_gold_indices = set()
    matched_count = 0

    for p_call in pred_calls:
        for g_idx, g_call in enumerate(gold_calls):
            if g_idx not in matched_gold_indices and call_matches(p_call, g_call):
                matched_gold_indices.add(g_idx)
                matched_count += 1
                break

    M = matched_count
    total_gold = len(gold_calls)
    U = len(pred_calls) - M  # Unmatched submitted calls (including rejected calls)

    precision = M / (M + U) if (M + U) > 0 else 0.0
    recall = M / total_gold if total_gold > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    state_match = (pred_hash == gold_hash)

    # Core condition: state_hash must equal gold_hash AND M > 0, otherwise score = 0
    if state_match and M > 0:
        final_sample_score = f1
    else:
        final_sample_score = 0.0

    return {
        "score": final_sample_score,
        "state_match": state_match,
        "matched_calls": M,
        "gold_calls": total_gold,
        "unmatched_calls": U,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

