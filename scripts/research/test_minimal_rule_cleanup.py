import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json
import sys
from pathlib import Path

WORKSPACE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from lifemate.evaluation.metrics import compute_sample_score, call_matches
from lifemate.adapters.sandbox.zaic_env_adapter import ZaicSandboxEnvFactory
from lifemate.adapters.storage.jsonl_dataset_reader import JsonlDatasetReader

# 1. Load Catalog to know schemas, required fields, and enum[0]
catalog = json.load(open(DATA_ROOT / "train" / "train_tools_catalog.json", encoding="utf-8"))
tool_schemas = {}
for app in catalog.get("apps", []):
    for tool in app.get("tools", []):
        tool_schemas[tool["name"]] = tool

with open(DISTILLATION_DIR / "master_train_distilled.jsonl", "r", encoding="utf-8") as f:
    master_records = [json.loads(line) for line in f if line.strip() and json.loads(line).get("state_match", False)]

print(f"Total State Match records to test: {len(master_records)}")

# Measure baseline scores before post-clean
orig_scores = [r["score"] for r in master_records]
orig_f1s = [r["f1"] for r in master_records]
orig_perfect = sum(1 for s in orig_scores if s == 1.0)

print(f"Baseline Average Score: {sum(orig_scores)/len(orig_scores)*100:.2f}%")
print(f"Baseline Average F1:    {sum(orig_f1s)/len(orig_f1s)*100:.2f}%")
print(f"Baseline Perfect 1.00:  {orig_perfect}/{len(master_records)} ({orig_perfect/len(master_records)*100:.2f}%)")

def clean_arguments(tool_name: str, args: dict) -> dict:
    cleaned = dict(args)
    schema = tool_schemas.get(tool_name)
    if not schema:
        return cleaned
    
    params_schema = schema.get("parameters", {}).get("properties", {})
    required_keys = set(schema.get("parameters", {}).get("required", []))
    
    # Rule A: Unfiltered read tools -> {}
    action_part = tool_name.split(".")[-1]
    if action_part.startswith("list_") or action_part.startswith("get_status") or action_part == "guide" or action_part == "list":
        return {}
        
    for k, v in list(cleaned.items()):
        if k not in required_keys:
            prop = params_schema.get(k, {})
            # Rule B: If enum and value == enum[0], strip it
            if "enum" in prop and prop["enum"]:
                if v == prop["enum"][0]:
                    del cleaned[k]
                    continue
            # Rule C: Optional label/note/description on write tools if not strictly required
            if k in ["label", "note", "description", "category", "repeat", "title", "reason"]:
                # If repeat is once, strip
                if k == "repeat" and v == "once":
                    del cleaned[k]
                    continue
                # If alarm label or transfer note
                if tool_name in ["alarm.create", "ewallet.transfer", "banking.transfer"]:
                    if k in ["label", "note"]:
                        del cleaned[k]
                        continue

    return cleaned

cleaned_scores = []
cleaned_f1s = []
cleaned_perfect = 0

diff_count = 0
for r in master_records:
    golds = r["gold_tool_calls"]
    preds = r["predicted_tool_calls"]
    pred_hash = r["predicted_state_hash"]
    gold_hash = r["gold_state_hash"]
    
    # Apply cleaning to predicted calls
    cleaned_preds = []
    has_diff = False
    for c in preds:
        c_name = c["name"]
        c_args = c.get("arguments", {})
        c_clean = clean_arguments(c_name, c_args)
        if c_clean != c_args:
            has_diff = True
        cleaned_preds.append({"name": c_name, "arguments": c_clean})
        
    if has_diff:
        diff_count += 1
        
    # Re-evaluate score
    eval_res = compute_sample_score(cleaned_preds, golds, pred_hash, gold_hash)
    cleaned_scores.append(eval_res["score"])
    cleaned_f1s.append(eval_res["f1"])
    if eval_res["score"] == 1.0:
        cleaned_perfect += 1

print("\n" + "=" * 80)
print(f"AFTER APPLYING MINIMAL RULE & ENUM[0] CLEANUP:")
print(f"Samples affected by cleanup: {diff_count}/{len(master_records)} ({diff_count/len(master_records)*100:.1f}%)")
print(f"New Average Score:           {sum(cleaned_scores)/len(cleaned_scores)*100:.2f}% (was {sum(orig_scores)/len(orig_scores)*100:.2f}%)")
print(f"New Average F1:              {sum(cleaned_f1s)/len(cleaned_f1s)*100:.2f}% (was {sum(orig_f1s)/len(orig_f1s)*100:.2f}%)")
print(f"New Perfect 1.00 Count:      {cleaned_perfect}/{len(master_records)} ({cleaned_perfect/len(master_records)*100:.2f}%) (was {orig_perfect})")
print("=" * 80)

# Check train_0028 specifically
for r in master_records:
    if r["id"] == "train_0028":
        preds = [{"name": c["name"], "arguments": clean_arguments(c["name"], c.get("arguments", {}))} for c in r["predicted_tool_calls"]]
        res = compute_sample_score(preds, r["gold_tool_calls"], r["predicted_state_hash"], r["gold_state_hash"])
        print(f"\ntrain_0028 after cleanup:")
        print(f"  Preds: {preds}")
        print(f"  Gold:  {r['gold_tool_calls']}")
        print(f"  Score: {res['score']} | F1: {res['f1']} | State Match: {res['state_match']}")
